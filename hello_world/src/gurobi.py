import gurobipy as gp
from gurobipy import GRB


def optimize(ffs, global_optimize=True):
    def cityblock_variable(model, v1, v2, bias, weight, intercept):
        # delta_x, delta_y = model.addVar(lb=-GRB.INFINITY), model.addVar(lb=-GRB.INFINITY)
        abs_delta_x, abs_delta_y = model.addVar(), model.addVar()
        # model.addLConstr(delta_x == v1[0] - v2[0])
        # model.addLConstr(delta_y == v1[1] - v2[1])
        delta_x = v1[0] - v2[0]
        delta_y = v1[1] - v2[1]
        model.addLConstr(abs_delta_x >= delta_x)
        model.addLConstr(abs_delta_x >= -delta_x)
        model.addLConstr(abs_delta_y >= delta_y)
        model.addLConstr(abs_delta_y >= -delta_y)
        cityblock_distance = model.addVar(lb=-GRB.INFINITY)
        model.addLConstr(
            cityblock_distance == weight * (abs_delta_x + abs_delta_y + bias) + intercept
        )
        return cityblock_distance

    print("Optimizing...")
    # k = [
    #     ff
    #     for ff in self.get_ffs()
    #     if any([self.get_origin_pin(curpin).slack < 0 for curpin in ff.dpins])
    # ]
    with gp.Env(empty=True) as env:
        env.setParam("LogToConsole", 1)
        env.start()

        def solve(optimize_ffs):
            with gp.Model(env=env) as model:
                for ff in optimize_ffs:
                    ff_vars[ff.name] = model.addVar(name=ff.name + "0"), model.addVar(
                        name=ff.name + "1"
                    )
                model.setParam("OutputFlag", 0)

                # model.setParam(GRB.Param.Presolve, 2)
                # if global_optimize:
                # else:
                #     # optimize_ffs_names = [ff.name for ff in optimize_ffs]
                #     # print(optimize_ffs_names)
                #     # for ff in self.get_ffs():
                #     #     if ff.name in optimize_ffs_names:
                #     #         ff_vars[ff.name] = model.addVar(name=ff.name + "0"), model.addVar(
                #     #             name=ff.name + "1"
                #     #         )
                #     # else:
                #     #     ff_vars[ff.name] = ff.pos
                #     for ff in optimize_ffs:
                #         ff_vars[ff.name] = model.addVar(name=ff.name + "0"), model.addVar(
                #             name=ff.name + "1"
                #         )

                # dis2ori_locations = []
                negative_slack_vars = []
                for ff in optimize_ffs:
                    for curpin in ff.dpins:
                        ori_slack = self.get_origin_pin(curpin).slack
                        prev_pin = self.get_prev_pin(curpin)
                        prev_pin_displacement_delay = 0
                        if prev_pin:
                            current_pin = self.get_pin(curpin)
                            current_pin_pos = [
                                a + b
                                for a, b in zip(ff_vars[current_pin.inst.name], current_pin.rel_pos)
                            ]
                            dpin_pin = self.get_pin(prev_pin)
                            dpin_pin_pos = dpin_pin.pos
                            ori_distance = self.original_pin_distance(prev_pin, curpin)
                            prev_pin_displacement_delay = cityblock_variable(
                                model,
                                current_pin_pos,
                                dpin_pin_pos,
                                -ori_distance,
                                self.setting.displacement_delay,
                                0,
                            )
                            # prev_pin_displacement_delay = model.addVar()

                        displacement_distances = []
                        prev_ffs = self.get_prev_ffs(curpin)
                        for qpin, pff in prev_ffs:
                            pff_pin = self.get_pin(pff)
                            qpin_pin = self.get_pin(qpin)
                            pff_pos = [
                                a + b for a, b in zip(ff_vars[pff_pin.inst.name], pff_pin.rel_pos)
                            ]
                            if qpin_pin.is_ff:
                                qpin_pos = [
                                    a + b
                                    for a, b in zip(ff_vars[qpin_pin.inst.name], qpin_pin.rel_pos)
                                ]
                            else:
                                qpin_pos = qpin_pin.pos
                            ori_distance = self.original_pin_distance(pff, qpin)
                            distance_var = cityblock_variable(
                                model,
                                pff_pos,
                                qpin_pos,
                                -ori_distance,
                                self.setting.displacement_delay,
                                -self.qpin_delay_loss(pff),
                            )
                            # distance_var = model.addVar()
                            displacement_distances.append(distance_var)
                        min_displacement_distance = 0
                        if len(displacement_distances) > 0:
                            min_displacement_distance = model.addVar(lb=-GRB.INFINITY)
                            model.addConstr(
                                min_displacement_distance == gp.min_(displacement_distances)
                            )

                        slack_var = model.addVar(name=curpin, lb=-GRB.INFINITY)
                        model.addConstr(
                            slack_var
                            == ori_slack - (prev_pin_displacement_delay + min_displacement_distance)
                        )
                        negative_slack_var = model.addVar(
                            name=f"negative_slack for {curpin}", lb=-GRB.INFINITY
                        )
                        model.addConstr(negative_slack_var == gp.min_(slack_var, 0))
                        negative_slack_vars.append(negative_slack_var)

                    # dis2ori = cityblock_variable(model, ff_vars[ff.name], ff.pos, 0, 1, 0)
                    # dis2ori_locations.append(dis2ori)

                model.setObjective(-gp.quicksum(negative_slack_vars))
                # model.setObjectiveN(-gp.quicksum(min_negative_slack_vars), 0, priority=1)
                # model.setObjectiveN(gp.quicksum(dis2ori_locations), 1, priority=0)
                model.optimize()
                for ff in optimize_ffs:
                    name = ff.name
                    new_pos = (ff_vars[name][0].X, ff_vars[name][1].X)
                    new_pos = np.int32(new_pos)
                    self.get_ff(name).moveto(new_pos)
                # for name, ff_var in ff_vars.items():
                #     if isinstance(ff_var[0], float):
                #         self.get_ffs(name)[0].moveto((ff_var[0], ff_var[1]))
                #     else:
                #         self.get_ffs(name)[0].moveto((ff_var[0].X, ff_var[1].X))
                #     ff_vars[name] = self.get_ff(name).pos

        ff_vars = self.get_static_vars()
        if not global_optimize:
            pin_list = self.get_end_ffs()
            # pin_name = pin_list[0]
            ff_paths = set()
            ffs_calculated = set()
            ff_path_all = [self.get_ff_path(pin_name) for pin_name in pin_list]
            ff_path_all.sort(key=lambda x: len(x), reverse=True)
            for ff_path in tqdm(ff_path_all):
                ff_paths.update(ff_path)
                # if len(ff_paths) < 50:
                #     ff_paths.update(ff_path)
                #     continue
                solve([self.get_ff(pin_name) for pin_name in (ff_paths - ffs_calculated)])
                for name in ff_paths:
                    ff_vars[name] = self.get_ff(name).pos
                ffs_calculated.update(ff_paths)
                ff_paths.clear()
        else:
            solve(self.get_ffs())
