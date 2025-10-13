use crate::*;
use std::collections::BTreeSet;
use std::fmt::Debug;

// The item stored in the BTreeSet for sorting: (Reverse<P>, ID)
// ID is used for deterministic tie-breaking when P is equal.
type Item<P> = (Reverse<P>, usize);

pub struct TopKRecorder<P, K>
where
    P: Ord + Clone + Debug,
    K: Eq + Hash + Clone + Debug,
{
    k: usize,
    // 1. Stores the TOP K elements: (Reverse<P>, ID). Max P is at the front.
    top_k: BTreeSet<Item<P>>,

    // 2. STORES ALL ELEMENTS: Maps actual Key K to its (Priority, ID).
    master_data: Dict<K, (P, usize)>,

    // 3. Counter for generating new unique IDs.
    id_counter: usize,
}

impl<P, K> TopKRecorder<P, K>
where
    P: Ord + Clone + Debug,
    K: Eq + Hash + Clone + Debug,
{
    pub fn new(k: usize) -> Self {
        TopKRecorder {
            k,
            top_k: BTreeSet::new(),
            master_data: Dict::new(),
            id_counter: 0,
        }
    }

    /// Access the maximum element in O(1) time.
    pub fn max_element(&self) -> Option<(P, K)> {
        self.top_k.iter().next().and_then(|(_, id)| {
            self.master_data
                .values()
                .find(|(_, stored_id)| stored_id == id)
                .map(|(p, _)| {
                    (
                        p.clone(),
                        self.master_data
                            .keys()
                            .find(|k| self.master_data[*k].1 == *id)
                            .unwrap()
                            .clone(),
                    )
                })
        })
    }

    /// Records a new element or updates the priority of an existing one.
    /// Time complexity: O(log k) normally, O(N) during eviction/refill.
    pub fn record(&mut self, new_priority: P, key: K) {
        // Retrieve old data or assign a new ID
        let (old_priority, old_id) = match self.master_data.get(&key) {
            Some((p, id)) => (Some(p.clone()), *id),
            None => {
                // Assign a new ID for the new key
                self.id_counter += 1;
                (None, self.id_counter)
            }
        };

        // 1. PRE-UPDATE: If the key was tracked, remove its old representation from top_k (if it was there).
        let was_in_top_k = if let Some(old_p) = old_priority.as_ref() {
            let old_item = (Reverse(old_p.clone()), old_id);
            self.top_k.remove(&old_item) // returns true if the item was present
        } else {
            false
        };

        // 2. UPDATE MASTER MAP: Store the new priority and the (potentially new) ID.
        self.master_data
            .insert(key.clone(), (new_priority.clone(), old_id));

        // 3. ATTEMPT INSERTION AND EVICTION
        let current_min_priority = self
            .top_k
            .iter()
            .next_back() // Gets the minimum element (last in Max-Heap)
            .map(|(rev_p, _)| rev_p.0.clone());

        let new_item = (Reverse(new_priority.clone()), old_id);

        // Check if the new item qualifies.
        let qualifies = (self.top_k.len() < self.k && !was_in_top_k)
            || new_priority > *current_min_priority.as_ref().map_or(&new_priority, |p| p);

        let mut needs_refill = false;

        if qualifies {
            // If set is full AND new item is better, remove the minimum (the last item)
            if self.top_k.len() == self.k {
                // We know it's full, and the new item is better, so pop the smallest.
                self.top_k.pop_last();
            }

            // Insert the new item.
            self.top_k.insert(new_item);
        } else if was_in_top_k {
            // Priority dropped below the minimum. It was removed in step 1 and didn't qualify
            // to be re-inserted here. The set size is now k-1.
            needs_refill = true;
        }

        // 4. CHECK FOR REFILL
        if needs_refill {
            self.refill_top_k();
        }
    }

    /// Finds and inserts the next best element from the full set of tracked priorities
    /// to restore the top_k set to size K. This is an O(N) operation.
    fn refill_top_k(&mut self) {
        if self.top_k.len() == self.k {
            return;
        }

        // Initialize best_candidate to store (Priority, ID)
        let mut best_candidate: Option<(P, usize)> = None;

        // --- O(N) ITERATION OVER ALL ELEMENTS (master_data) ---
        for (_, (priority, id)) in self.master_data.iter() {
            let item = (Reverse(priority.clone()), *id);

            // 1. Skip elements already in the top_k set (O(log k) check).
            if self.top_k.contains(&item) {
                continue;
            }

            // 2. Track the best candidate found so far among the "rest", using ID for tie-breaking.
            match best_candidate.as_ref() {
                None => {
                    // First candidate found
                    best_candidate = Some((priority.clone(), *id));
                }
                Some((best_p, best_id)) => {
                    // Compare the current candidate (priority, id) against the best found so far
                    if priority > best_p {
                        // Current is strictly better priority
                        best_candidate = Some((priority.clone(), *id));
                    } else if priority == best_p && id < best_id {
                        // Priorities are tied. Lower ID is considered "better" (since ID is the secondary sort key).
                        best_candidate = Some((priority.clone(), *id));
                    }
                }
            }
        }

        // --- Insert the best candidate found, if any ---
        if let Some((p, id)) = best_candidate {
            self.top_k.insert((Reverse(p), id));
        }
    }

    /// Returns the top k elements and their priorities, sorted descendingly by priority.
    pub fn top_k(&self) -> Vec<(P, K)> {
        // Collects results from the BTreeSet and looks up the K key in the master map.
        self.top_k
            .iter()
            .map(|(_, id)| {
                // Find the key K associated with this ID in the master map (O(N) search on values, sadly)
                let (p, k) = self
                    .master_data
                    .iter()
                    .find(|(_, (_, stored_id))| stored_id == id)
                    .map(|(k, (p, _))| (p.clone(), k.clone()))
                    .unwrap(); // Should always succeed if the ID is in top_k

                (p, k)
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_top_k_recorder_basic() {
        // Set K to 3 for the test
        let k = 3;
        let mut recorder = TopKRecorder::<i32, String>::new(k);
        // --- Phase 1: Initial Filling (Size < K) ---
        recorder.record(10, "A".to_string());
        assert!(recorder.top_k() == vec![(10, "A".to_string())]);

        // --- Phase 2: Maintaining K (Changing Priority) ---
        recorder.record(20, "A".to_string());
        assert!(recorder.top_k() == vec![(20, "A".to_string())]);

        // --- Phase 3: Inserting a worse element ---
        recorder.record(15, "B".to_string());
        assert!(recorder.top_k() == vec![(20, "A".to_string()), (15, "B".to_string())]);

        // --- Phase 4: Inserting a better element ---
        recorder.record(25, "C".to_string());
        assert!(
            recorder.top_k()
                == vec![(25, "C".to_string()), (20, "A".to_string()), (15, "B".to_string())]
        );

        recorder.record(2, "D".to_string());
        recorder.record(3, "E".to_string());
        recorder.record(1, "A".to_string());
        assert!(
            recorder.top_k()
                == vec![(25, "C".to_string()), (15, "B".to_string()), (3, "E".to_string())]
        );

    }

    #[test]
    fn test_top_k_recorder_with_id_refill() {
        // Set K to 3 for the test
        let k = 3;
        let mut recorder = TopKRecorder::<i32, String>::new(k);
        // --- Phase 1: Initial Filling (Size < K) ---
        println!("--- Phase 1: Initial Filling ---");
        recorder.record(10, "A".to_string());
        recorder.record(50, "B".to_string());
        recorder.record(20, "C".to_string());

        // Expected: [('B', 50), ('C', 20), ('A', 10)]
        assert!(
            recorder.top_k()
                == vec![
                    (50, "B".to_string()),
                    (20, "C".to_string()),
                    (10, "A".to_string())
                ]
        );

        // --- Phase 2: Maintaining K (Inserting a better element) ---
        println!("\n--- Phase 2: Replacing the minimum ---");
        // Element D (priority 60) should replace the current minimum, A (priority 10).
        recorder.record(60, "D".to_string());

        // Expected: [('D', 60), ('B', 50), ('C', 20)]. 'A' (10) is gone.
        assert!(
            recorder.top_k()
                == vec![
                    (60, "D".to_string()),
                    (50, "B".to_string()),
                    (20, "C".to_string())
                ]
        );

        // --- Phase 3: Inserting a worse element (should be ignored) ---
        println!("\n--- Phase 3: Inserting a worse element ---");
        // Element E (priority 15) is worse than the current minimum, C (priority 20).
        // The set should remain unchanged.
        recorder.record(15, "E".to_string());
        assert!(
            recorder.top_k()
                == vec![
                    (60, "D".to_string()),
                    (50, "B".to_string()),
                    (20, "C".to_string())
                ]
        );

        // --- Phase 4: Inserting an existing element (should update its priority) ---
        println!("\n--- Phase 4: Inserting an existing element ---");
        // Element C (priority 10) should replace the current minimum, C (priority 20).
        // The last element should be E (priority 15), not C (priority 10).
        recorder.record(10, "C".to_string());
        assert!(
            recorder.top_k()
                == vec![
                    (60, "D".to_string()),
                    (50, "B".to_string()),
                    (15, "E".to_string())
                ]
        );

        // --- Phase 5: Inserting an existing element to become the new minimum ---
        recorder.record(15, "D".to_string());
        assert!(
            recorder.top_k()
                == vec![
                    (50, "B".to_string()),
                    (15, "D".to_string()),
                    (15, "E".to_string())
                ]
        );
    }
}
