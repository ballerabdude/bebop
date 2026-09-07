//! Operator navigation goal (navd) — shared state + change broadcast.
//!
//! The app sends `SetNavigationGoal`; the WS handler stores it here and
//! every connected client (notably the bebop-vision goal-drive process)
//! receives the change as a `NavigationGoalState` server push. State also
//! flushes once to each client right after connect so late joiners see
//! the current goal.
//!
//! Ownership stays with the operator UI: any client can set or clear.
//! Safety is unchanged — the goal only steers the goal-drive planner,
//! whose twists still pass the supervisor's mode/deadman/E-stop gates.

use bebop_proto::runtime::v1 as proto;
use std::sync::RwLock;
use tokio::sync::broadcast;

/// Current goal, mirroring the `SetNavigationGoal` semantics.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum NavGoal {
    /// No active goal (robot holds / operator drives).
    None,
    /// Body-frame heading offset from current pose (rad, + left).
    Heading(f32),
    /// Destination point in the odometry frame (m).
    PointOdom(f32, f32),
}

impl NavGoal {
    pub fn to_proto(self) -> proto::NavigationGoalState {
        match self {
            NavGoal::None => proto::NavigationGoalState {
                active: false,
                goal: None,
            },
            NavGoal::Heading(rad) => proto::NavigationGoalState {
                active: true,
                goal: Some(proto::navigation_goal_state::Goal::HeadingRad(rad)),
            },
            NavGoal::PointOdom(x, y) => proto::NavigationGoalState {
                active: true,
                goal: Some(proto::navigation_goal_state::Goal::PointOdom(proto::Vec2 {
                    x,
                    y,
                })),
            },
        }
    }
}

/// Shared current-goal slot + change broadcast.
pub struct NavGoalShared {
    state: RwLock<NavGoal>,
    tx: broadcast::Sender<NavGoal>,
}

impl Default for NavGoalShared {
    fn default() -> Self {
        Self::new()
    }
}

impl NavGoalShared {
    pub fn new() -> Self {
        let (tx, _) = broadcast::channel(16);
        Self {
            state: RwLock::new(NavGoal::None),
            tx,
        }
    }

    /// Store a new goal and notify subscribers. Any goal value is legal;
    /// the WS layer rejects malformed requests before reaching here.
    pub fn set(&self, goal: NavGoal) {
        *self.state.write().unwrap() = goal;
        let _ = self.tx.send(goal);
    }

    pub fn get(&self) -> NavGoal {
        *self.state.read().unwrap()
    }

    /// Subscribe to goal changes. The receiver immediately sees future
    /// changes only — the current state rides the post-connect flush.
    pub fn subscribe(&self) -> broadcast::Receiver<NavGoal> {
        self.tx.subscribe()
    }
}
