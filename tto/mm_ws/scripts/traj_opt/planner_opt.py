import os
import sys

sys.path.insert(0, "..")

import numpy as np


import optas
import casadi as cs
from scipy import interpolate

def interpolate_waypoints(waypoints, n, m, mode="cubic"):  # linear
    """
    Interpolate the waypoints using interpolation.
    """
    data = np.zeros([n, m])
    x = np.linspace(0, 1, waypoints.shape[0])
    for i in range(waypoints.shape[1]):
        y = waypoints[:, i]

        t = np.linspace(0, 1, n + 2)
        if mode == "linear":  # clamped cubic spline
            f = interpolate.interp1d(x, y, "linear")
        if mode == "cubic":  # clamped cubic spline
            f = interpolate.CubicSpline(x, y, bc_type="clamped")
        elif mode == "quintic":  # seems overkill to me
            pass
        data[:, i] = f(t[1:-1])  #
        # plt.plot(x, y, 'o', t[1:-1], data[:, i], '-') #  f(np.linspace(0, 1, 5 * n+2))
        # plt.show()
    return data

class TTOPlanner:
    def __init__(
        self,
        robot,
        link_ee,
        link_gripper,
        collision_avoidance=True,
        standoff_distance=-0.1,
        standoff_offset=-10,
        T=100,
        Tmax=10,
    ):

        # trajectory parameters
        self.T = T  # no. time steps in trajectory
        self.Tmax = Tmax  # trajectory of 5 secs
        # t = optas.linspace(0, self.Tmax, self.T)
        t = optas.linspace(0, self.T, self.Tmax)
        # self.dt = float((t[1] - t[0]).toarray()[0, 0])  # time step
        self.dt = 0.1  # time step
        self.standoff_offset = standoff_offset
        self.standoff_distance = standoff_distance

        # Setup robot
        self.robot = robot
        self.robot_name = robot.get_name()
        self.link_ee = link_ee
        self.link_gripper = link_gripper
        self.gripper_points = robot.surface_pc_map[link_gripper].points
   
        self.gripper_tf = robot.get_link_transform_function(
            link=link_gripper, base_link=link_ee
        )
        self.collision_avoidance = collision_avoidance

    def setup_optimization(self, ref_trajectory, use_standoff=False, axis_standoff="x"):
        # Setup optimization builder
        builder = optas.OptimizationBuilder(T=self.T, robots=[self.robot])

        # Setup parameters
        qc = builder.add_parameter("qc", self.robot.ndof)  # Current joint configuration
        sdf_cost_all = builder.add_parameter("sdf_cost_all", self.robot.field_size)
        sdf_cost_obstacle = builder.add_parameter(
            "sdf_cost_obstacle", self.robot.field_size
        )
        base_position = builder.add_parameter("base_position", 3)

        # Constraint: initial configuration
        builder.initial_configuration(
            self.robot_name,
            self.robot.extract_optimized_dimensions(qc),
        )
        builder.initial_configuration(
            self.robot_name, time_deriv=1
        )  # Initial joint velocity is zero

        # Constraint: dynamics
        builder.integrate_model_states(
            self.robot_name,
            time_deriv=1,  # Integrate velocities to positions
            dt=self.dt,
        )

        # Get joint trajectory
        Q = builder.get_robot_states_and_parameters(self.robot_name)

        self.fk = self.robot.get_global_link_transform_function(
            self.link_ee, n=self.T
        )

        tf_gripper = self.fk(Q)
        # Cost: Track the reference trajectory
        cost_tracking = cs.MX.zeros(1)
        
        scale = 150

        # ///////////////////   6dof ////////////////////////////////////////////////
        for t in range(self.T):
            RT_current = tf_gripper[t]
            gripper_mesh_pts_RT = RT_current[
                :3, :3
            ] @ self.gripper_points.T + RT_current[:3, 3].reshape((3, 1))
            RT_ref = ref_trajectory[t]  # Reference position at timestep t
            gripper_mesh_pts_RTref = RT_ref[:3, :3] @ self.gripper_points.T + RT_ref[
                :3, 3
            ].reshape((3, 1))

            cost_tracking += optas.sumsqr(
                gripper_mesh_pts_RT - gripper_mesh_pts_RTref
            )  # Minimize position error

        builder.add_cost_term("tracking_cost", scale * cost_tracking)

        # Obstacle avoidance
        if self.collision_avoidance:
            points_world = None
            for t in range(self.T):
                q_t = Q[:, t]
                for name in self.robot.surface_pc_map.keys():
                    tf = self.robot.visual_tf[name](q_t)
                    points = self.robot.surface_pc_map[name].points
                    points_world_t = (
                        tf[:3, :3] @ points.T
                        + tf[:3, 3].reshape((3, 1))
                        + base_position.reshape((3, 1))
                    )
                    if points_world is None:
                        points_world = points_world_t
                    else:
                        points_world = optas.horzcat(points_world, points_world_t)
            offsets = self.robot.points_to_offsets(points_world.T)

            scale = 0.2  # deafult 10

            builder.add_cost_term(
                "cost_obstacle", scale * optas.sumsqr(sdf_cost_obstacle[offsets])
            )


        # Cost: Minimize joint velocity
        dQ = builder.get_robot_states_and_parameters(self.robot_name, time_deriv=1)
        builder.add_cost_term("min_joint_vel", 0.01 * optas.sumsqr(dQ))

        # Constraint: Joint position limits
        builder.enforce_model_limits(self.robot_name)

        # Setup solver
        solver_options = {"ipopt": {"max_iter": 120, "tol": 1e-15}}
        self.solver = optas.CasADiSolver(builder.build()).setup(
            "ipopt", solver_options=solver_options
        )

    def plan(
        self,
        qc,
        RT,
        sdf_cost_obstacle,
        base_position,
        q_solution=None,
        use_standoff=True,
        axis_standoff="x",
    ):
        self.setup_optimization(
            goal_size=1, use_standoff=use_standoff, axis_standoff=axis_standoff
        )
        tf_goal = np.zeros((16, 1))
        tf_goal[:, 0] = RT.flatten()

        # Set initial seed, note joint velocity will be set to zero
        if q_solution is None:
            Q0 = optas.diag(qc) @ optas.DM.ones(self.robot.ndof, self.T)
        else:
            # interpolate waypoints
            data = interpolate_waypoints(
                np.stack([qc, q_solution]), self.T, self.robot.ndof
            )
            index = np.array(self.robot.parameter_joint_indexes).astype(np.int32)
            data[:, index] = np.array(qc)[index]
            Q0 = optas.DM(data.T)

        self.solver.reset_initial_seed(
            {f"{self.robot_name}/q/x": self.robot.extract_optimized_dimensions(Q0)}
        )

        # Set parameters
        self.solver.reset_parameters(
            {
                "qc": optas.DM(qc),
                "tf_goal": optas.DM(tf_goal),
                "sdf_cost_obstacle": optas.DM(sdf_cost_obstacle),
                "base_position": optas.DM(base_position),
                f"{self.robot_name}/q/p": self.robot.extract_parameter_dimensions(Q0),
            }
        )

        # Solve problem
        solution = self.solver.solve()

        # Get robot configuration
        Q = solution[f"{self.robot_name}/q"]
        dQ = solution[f"{self.robot_name}/dq"]
        cost = solution["f"]
        return Q.toarray(), dQ.toarray(), cost.toarray().flatten()

    def plan_goalset(
        self,
        qc,
        RTs,
        sdf_cost_all,
        sdf_cost_obstacle,
        base_position,
        q_solutions=None,
        use_standoff=True,
        axis_standoff="x",
        interpolate=True,
    ):
        n = RTs.shape[0]
        self.setup_optimization(
            goal_size=n, use_standoff=use_standoff, axis_standoff=axis_standoff
        )
        tf_goal = np.zeros((16, n))
        for i in range(n):
            RT = RTs[i]
            tf_goal[:, i] = RT.flatten()

        if q_solutions is None:
            Q0 = optas.diag(qc) @ optas.DM.ones(self.robot.ndof, self.T)
        else:
            # intialize the goal
            cost_all = []
            dist_all = []
            plan_all = []
            for i in range(q_solutions.shape[1]):
                q_solution = q_solutions[:, i]
                # interpolate waypoints
                data = interpolate_waypoints(
                    np.stack([qc, q_solution]), self.T, self.robot.ndof
                )
                index = np.array(self.robot.parameter_joint_indexes).astype(np.int32)
                data[:, index] = np.array(qc)[index]
                plan = data.T
                plan_all.append(plan.copy())
                cost, dist = self.robot.compute_plan_cost(
                    plan, sdf_cost_obstacle, base_position
                )
                # print(f"plan {i}, cost {cost:.2f}, dist {dist:.2f}")
                cost_all.append(cost)
                dist_all.append(dist)  # large cost reach is better
            ind = np.lexsort((dist_all, cost_all))  # sort by cost, then by distance
            print("intialize with solution", ind[0])
            if interpolate:
                Q0 = optas.DM(plan_all[ind[0]])
            else:
                Q0 = optas.diag(qc) @ optas.DM.ones(self.robot.ndof, self.T)
                for i in range(self.T + self.standoff_offset, self.T):
                    Q0[:, i] = plan_all[ind[0]][:, self.T - 1]

        # Set initial seed, note joint velocity will be set to zero
        self.solver.reset_initial_seed(
            {f"{self.robot_name}/q/x": self.robot.extract_optimized_dimensions(Q0)}
        )

        # Set parameters
        self.solver.reset_parameters(
            {
                "qc": optas.DM(qc),
                "tf_goal": optas.DM(tf_goal),
                "sdf_cost_all": optas.DM(sdf_cost_all),
                "sdf_cost_obstacle": optas.DM(sdf_cost_obstacle),
                "base_position": optas.DM(base_position),
                f"{self.robot_name}/q/p": self.robot.extract_parameter_dimensions(Q0),
            }
        )

        # Solve problem
        solution = self.solver.solve()

        # Get robot configuration
        Q = solution[f"{self.robot_name}/q"]
        dQ = solution[f"{self.robot_name}/dq"]
        cost = solution["f"]
        return Q.toarray(), dQ.toarray(), cost.toarray().flatten()

    def plan_ref_trajectory(
        self, qc, ref_trajectory, sdf_cost_obstacle, base_position, interpolate=True
    ):
        # Setup optimization for tracking the reference trajectory
        self.setup_optimization(ref_trajectory)

        # Initialize trajectory (Q0)
        if interpolate:
            q_target = qc.copy()
            Q0 = interpolate_waypoints(
                np.stack([qc, q_target]), self.T, self.robot.ndof
            ).T
        else:
            Q0 = optas.diag(qc) @ optas.DM.ones(self.robot.ndof, self.T)

        # Set initial seed
        self.solver.reset_initial_seed(
            {f"{self.robot_name}/q/x": self.robot.extract_optimized_dimensions(Q0)}
        )

        # Set parameters
        self.solver.reset_parameters(
            {
                "qc": optas.DM(qc),
                "sdf_cost_obstacle": optas.DM(sdf_cost_obstacle),
                "base_position": optas.DM(base_position),
                f"{self.robot_name}/q/p": self.robot.extract_parameter_dimensions(Q0),
            }
        )

        # Solve optimization
        solution = self.solver.solve()

        # Extract results
        Q = solution[f"{self.robot_name}/q"]
        dQ = solution[f"{self.robot_name}/dq"]
        cost = solution["f"]
        return Q.toarray(), dQ.toarray(), cost.toarray().flatten()