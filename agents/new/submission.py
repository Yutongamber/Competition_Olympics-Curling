
class Env:

    def stepPhysics(self, actions_list, step = None):
        assert len(actions_list) == self.agent_num, print("The number of action needs to match with the number of agents!")
        self.actions_list = actions_list

        #current pos and v
        temp_pos_container = [self.agent_pos[i] for i in range(self.agent_num)]
        temp_v_container = [self.agent_v[i] for i in range(self.agent_num)]
        temp_a_container = self.actions_to_accel(actions_list)
        self.agent_accel = temp_a_container

        remaining_t = self.tau
        ignore_wall = copy.copy(self.global_wall_ignore)
        ignore_circle = copy.copy(self.global_circle_ignore)
        self.global_wall_ignore, self.global_circle_ignore = [], []   #only inherit once


        while True:
            if self.print_log:
                print('Remaining time = ', remaining_t)
                print('The pos = {}, the v = {}'.format(temp_pos_container, temp_v_container))

            earliest_wall_col_t, collision_wall_target, target_wall_idx, current_agent_idx = \
                self.bounceable_wall_collision_time(temp_pos_container, temp_v_container, remaining_t, ignore_wall)    #collision detection with walls

            earliest_circle_col_t, collision_circle_target, current_circle_idx, target_circle_idx = \
                self.circle_collision_time(temp_pos_container, temp_v_container, remaining_t, ignore_circle)

            if self.print_log:
                print('Wall t = {}, collide = {}, agent_idx = {}, wall_idx = {}'.format(
                    earliest_wall_col_t, collision_wall_target, current_agent_idx, target_wall_idx))
                print('Circle t = {}, collide = {}, agent_idx = {}, target_idx = {}'.format(
                    earliest_circle_col_t, collision_circle_target, current_circle_idx, target_circle_idx))


            if collision_wall_target is not None and collision_circle_target is None:
                if self.print_log:
                    print('HIT THE WALL!')

                temp_pos_container, temp_v_container, remaining_t, ignore_wall = \
                    self.handle_wall(target_wall_idx, collision_wall_target, current_agent_idx, earliest_wall_col_t,
                                     temp_pos_container, temp_v_container, remaining_t, ignore_wall)

            elif collision_wall_target is None and collision_circle_target == 'circle':
                if self.print_log:
                    print('HIT THE BALL!')

                temp_pos_container, temp_v_container, remaining_t, ignore_circle = \
                    self.handle_circle(target_circle_idx, collision_circle_target, current_circle_idx,
                                       earliest_circle_col_t, temp_pos_container, temp_v_container, remaining_t,
                                       ignore_circle)

            elif collision_wall_target is not None and collision_circle_target == 'circle':
                if self.print_log:
                    print('HIT BOTH!')

                if earliest_wall_col_t < earliest_circle_col_t:
                    if self.print_log:
                        print('PROCESS WALL FIRST!')

                    temp_pos_container, temp_v_container, remaining_t, ignore_wall = \
                        self.handle_wall(target_wall_idx, collision_wall_target, current_agent_idx, earliest_wall_col_t,
                                         temp_pos_container, temp_v_container, remaining_t, ignore_wall)

                elif earliest_wall_col_t >= earliest_circle_col_t:
                    if self.print_log:
                        print('PROCESS CIRCLE FIRST!')

                    temp_pos_container, temp_v_container, remaining_t, ignore_circle = \
                        self.handle_circle(target_circle_idx, collision_circle_target, current_circle_idx,
                                           earliest_circle_col_t, temp_pos_container, temp_v_container, remaining_t,
                                           ignore_circle)

                else:
                    raise NotImplementedError("collision error")

            else:   #no collision within this time interval
                if self.print_log:
                    print('NO COLLISION!')
                temp_pos_container, temp_v_container = self.update_all(temp_pos_container, temp_v_container, remaining_t, temp_a_container)
                break   #when no collision occurs, break the collision detection loop

        self.agent_pos = temp_pos_container
        self.agent_v = temp_v_container
        print("--------------------------")
        print("check pos: ", self.agent_pos)
        print("check v: ", self.agent_v)
        print("check actions: ", actions_list)
        print("duration: ", time.time()-time_s) # 0.002

    def circle_collision_time(self, pos_container, v_container, remaining_t, ignore):   #ignore = [[current_idx, target_idx, collision time]]


        #compute collision time between all circle
        current_idx = None
        target_idx = None
        current_min_t = remaining_t
        col_target = None

        for agent_idx in range(self.agent_num):
            pos1 = pos_container[agent_idx]
            v1 = v_container[agent_idx]
            m1 = self.agent_list[agent_idx].mass
            r1 = self.agent_list[agent_idx].r

            for rest_idx in range(agent_idx + 1, self.agent_num):
                pos2 = pos_container[rest_idx]
                v2 = v_container[rest_idx]
                m2 = self.agent_list[rest_idx].mass
                r2 = self.agent_list[rest_idx].r

                #compute ccd collision time
                collision_t = self.CCD_circle_collision(pos1, pos2, v1, v2, r1, r2, m1, m2, return_t=True)
                # print('agent {}, time on circle {} is  = {}, current_min_t = {}'.format(agent_idx,
                #                                                                         rest_idx,
                #                                                                         collision_t,
                #                                                                         remaining_t))
                # print('ignore list = ', ignore)

                if 0 <= collision_t < current_min_t:# and [agent_idx, rest_idx, collision_t] not in ignore:
                    current_min_t = collision_t
                    current_idx = agent_idx
                    target_idx = rest_idx
                    col_target = 'circle'

        return current_min_t, col_target, current_idx, target_idx

    
    def bounceable_wall_collision_time(self, pos_container, v_container, remaining_t, ignore):

        col_target = None
        col_target_idx = None
        current_idx = None
        current_min_t = remaining_t

        for agent_idx in range(self.agent_num):
            pos = pos_container[agent_idx]
            v = v_container[agent_idx]
            r = self.agent_list[agent_idx].r

            if v[0] == v[1] == 0:
                continue

            for object_idx in range(len(self.map['objects'])):
                object = self.map['objects'][object_idx]

                if object.can_pass():     #cross
                    continue
                if object.ball_can_pass and self.agent_list[agent_idx].type == 'ball':      #for table hockey game
                    continue

                #check the collision time and the collision target (wall and endpoint collision)
                temp_t, temp_col_target = object.collision_time(pos = pos, v = v, radius = r, add_info = [agent_idx, object_idx, ignore])

                if abs(temp_t) < 1e-10:   #the collision time computation has numerical error
                    temp_t = 0


                #if object_idx == 2:
                #    print('agent {}: time on wall {}({}) is = {}, current_min_t = {}'.format(
                #        agent_idx,object_idx,temp_col_target, temp_t, current_min_t))
                    #print('ignore list = ', ignore)

                if 0<= temp_t < current_min_t:
                    if temp_col_target == 'wall' or temp_col_target == 'arc':
                        check = ([agent_idx, object_idx, temp_t] not in ignore)
                    elif temp_col_target == 'l1' or temp_col_target == 'l2':
                        check = ([agent_idx, getattr(object, temp_col_target), temp_t] not in ignore)
                    else:
                        raise NotImplementedError('bounceable_wall_collision_time error')

                    if check:
                        current_min_t = temp_t
                        col_target = temp_col_target
                        col_target_idx = object_idx
                        current_idx = agent_idx

        return current_min_t, col_target, col_target_idx, current_idx
