#include "iar_dijkstra_planner/navfn.hpp"
#include "rclcpp/rclcpp.hpp"

namespace iar_dijkstra_planner
{
    NavFn::NavFn(int nx, int ny)
    {
          // create cell arrays
        RCLCPP_DEBUG(rclcpp::get_logger("rclcpp"), "[NavFn] Constructor");
        costarr_ = NULL;
        potarr_ = NULL;
        pending_ = NULL;
        setNavArr(nx, ny);

        curPotentArr_ = new int[BUFFERSIZE];
        nextPotentArr_ = new int[BUFFERSIZE];
        overflowPotentArr_ = new int[BUFFERSIZE];

        potentThreshInc_ = 2 * COST_NEUTRAL;


        npathbuf_ = npath_ = 0;
        pathx_ = pathy_ = NULL;
        
    }

    NavFn::~NavFn()
    {
        if (costarr_) {
            delete[] costarr_;
        }
        if (potarr_) {
            delete[] potarr_;
        }
        if (curPotentArr_){
            delete[] curPotentArr_;
        }
        if(nextPotentArr_){
            delete[] nextPotentArr_;
        }
        if(overflowPotentArr_){
            delete[] overflowPotentArr_;
        }
        if (pending_) {
            delete[] pending_;
        }

        if (pathx_) {
            delete[] pathx_;
        }
        if (pathy_) {
            delete[] pathy_;
        }
    }

    void NavFn::setNavArr(int nx, int ny)
    {
        RCLCPP_DEBUG(rclcpp::get_logger("rclcpp"), "[NavFn]: Array is %d x %d\n", nx, ny);

        nx_ = nx;
        ny_ = ny;
        ns_ = nx * ny;

        if (costarr_) {
            delete[] costarr_;
        }
        if (potarr_) {
            delete[] potarr_;
        }
        if (pending_) {
            delete[] pending_;
        }

        costarr_ = new COSTTYPE[ns_];  // cost array, 2d config space
        memset(costarr_, 0, ns_ * sizeof(COSTTYPE));
        potarr_ = new float[ns_];  // navigation potential array
        pending_ = new bool[ns_];
        memset(pending_, 0, ns_ * sizeof(bool));
        RCLCPP_DEBUG(rclcpp::get_logger("rclcpp"), "[NavFn]: Array is %d x %d\n", nx, ny);

    }


    void NavFn::setStart(int * start)
    {
        start_[0] = start[0];
        start_[1] = start[1];
    }
    void NavFn::setGoal(int * goal)
    {
        goal_[0] = goal[0];
        goal_[1] = goal[1];   
    }


    void
    NavFn::setCostmap(const COSTTYPE * cmap, bool allow_unknown)
    {
        COSTTYPE * cm = costarr_;
        for (int i = 0; i < ny_; i++) {
            int k = i * nx_;
            for (int j = 0; j < nx_; j++, k++, cmap++, cm++) {
                // This transforms the incoming cost values from NAV2::COSTMAP::2D
                //      NO_INFORMATION = 255;
                //      LETHAL_OBSTACLE = 254;
                //      INSCRIBED_INFLATED_OBSTACLE = 253;
                //      CostValue: 0 ~ 252

                // COST_UNKNOWN_ROS (255)               -> COST_OBS - 1 (253)
                // LETHAL_OBSTACLE (254)                -> COST_OBS (254)
                // INSCRIBED_INFLATED_OBSTACLE (253)    -> COST_OBS (254)
                // Values(0 ~ 252) -> values from COST_NEUTRAL (50) to max(COST_OBS_ROS (253), COST_NEUTRAL + COST_FACTOR * CostValue).
                *cm = COST_OBS; 
                int v = *cmap;
                if (v < COST_OBS_ROS) { // COST_OBS_ROS = 253
                    v = COST_NEUTRAL + COST_FACTOR * v;
                    if (v >= COST_OBS) {
                        v = COST_OBS - 1;
                    }
                    *cm = v;
                } else if (v == COST_UNKNOWN_ROS && allow_unknown) {
                    v = COST_OBS - 1;
                    *cm = v;
                }
            }
        }
    }

    bool NavFn::setNavFn()
    {
        // reset values in propagation arrays
        for (int i = 0; i < ns_; i++) {
            potarr_[i] = POT_HIGHEST;
        }

        // outer bounds of cost array
        COSTTYPE * pc;
        pc = costarr_;
        for (int i = 0; i < nx_; i++) {
            *pc++ = COST_OBS;
        }
        pc = costarr_ + (ny_ - 1) * nx_;
        for (int i = 0; i < nx_; i++) {
            *pc++ = COST_OBS;
        }
        pc = costarr_;
        for (int i = 0; i < ny_; i++, pc += nx_) {
            *pc = COST_OBS;
        }
        pc = costarr_ + nx_ - 1;
        for (int i = 0; i < ny_; i++, pc += nx_) {
            *pc = COST_OBS;
        }

        // priority buffers
        potentThresh_ = COST_OBS;
        n_curPotentArr_ = 0;
        n_nextPotentArr_ = 0;
        n_overflowPotentArr_ = 0;
        memset(pending_, 0, ns_ * sizeof(bool));

        // set start
        int k = start_[0] + start_[1] * nx_;
        potarr_[k] = 0;
        push_cur(k + 1); // left
        push_cur(k - 1); // right
        push_cur(k - nx_); // top
        push_cur(k + nx_); // bottom
        if (n_curPotentArr_ <= 0)
        {
            RCLCPP_WARN(
                rclcpp::get_logger("rclcpp"),
                "[NavFn] Planning Failed, as Starting cell sorrunded by Obstacles"
            );
            return false;
        }
            

        // find # of obstacle cells
        pc = costarr_;
        int ntot = 0;
        for (int i = 0; i < ns_; i++, pc++) {
            if (*pc >= COST_OBS) {
            ntot++;  // number of cells that are obstacles
            }
        }
        nobs_ = ntot;
        return true;
    }

    bool NavFn::propDijkstra(int cycles)
    {
        int max_blk_size = 0;  // max priority block size
        int n_cells = 0;  // number of cells put into priority blocks
        int cycle = 0;  // which cycle we're on

        int goalCell = goal_[1] * nx_ + goal_[0];
        bool propSuccess = false;
        for(; cycle < cycles; cycle++)
        {
            if (n_curPotentArr_ == 0 && n_nextPotentArr_ == 0)
            {
                break;
            }

            int i = n_curPotentArr_;
            int * pb = curPotentArr_;
            while(i-->0)
            {
                pending_[*(pb++)] = false;
            }

            pb = curPotentArr_;
            i = n_curPotentArr_;
            while(i-- >0)
            {
                updateCell(*pb++);
            }

            //stats
            n_cells += n_curPotentArr_;
            if (n_curPotentArr_>max_blk_size)
                max_blk_size = n_curPotentArr_;

            n_curPotentArr_ = n_nextPotentArr_;
            n_nextPotentArr_ = 0;
            pb =  curPotentArr_;
            curPotentArr_ = nextPotentArr_;
            nextPotentArr_ = pb;

            if (n_curPotentArr_ == 0){
                potentThresh_ += potentThreshInc_;
                n_curPotentArr_ = n_overflowPotentArr_;
                n_overflowPotentArr_ = 0;
                pb = curPotentArr_;
                curPotentArr_ = overflowPotentArr_;
                overflowPotentArr_ = pb;
            }   

            // Check we have hit the goal cell
            if(potarr_[goalCell] < POT_HIGHEST)
            {
                propSuccess = true;
                break;
            }
                
        }
        RCLCPP_DEBUG(
            rclcpp::get_logger("rclcpp"),
            "[NavFn] Used %d cycles, %d cells visited (%d%%), priority buf max %d\n",
            cycle, n_cells, (int)((n_cells * 100.0) / (ns_ - nobs_)), max_blk_size);    
        return propSuccess;
    }

    inline void NavFn::updateCell(int n)
    {
        // get neighbors
        float u, d, l, r;
        l = potarr_[n - 1];
        r = potarr_[n + 1];
        u = potarr_[n - nx_];
        d = potarr_[n + nx_];

        float minNeighborP;

        minNeighborP = std::min({l, r, u, d});
        if (costarr_[n] < COST_OBS)
        {
            float hf = static_cast<float>(costarr_[n]);
            float pot = minNeighborP + hf;

            if (pot < potarr_[n])
            {
                potarr_[n] = pot;
                if (pot < potentThresh_)
                {
                    if(l > pot + costarr_[n - 1]) {push_next(n - 1);}
                    if(r > pot + costarr_[n + 1]) {push_next(n + 1);}
                    if(u > pot + costarr_[n - nx_]) {push_next(n - nx_);}
                    if(d > pot + costarr_[n + nx_]) {push_next(n + nx_);}

                } else {
                    if(l > pot + costarr_[n - 1]) {push_over(n - 1);}
                    if(r > pot + costarr_[n + 1]) {push_over(n + 1);}
                    if(u > pot + costarr_[n - nx_]) {push_over(n - nx_);}
                    if(d > pot + costarr_[n + nx_]) {push_over(n + nx_);}
                }
            }
        }

    }

    int NavFn::calcPath(int n)
    {
        if (npathbuf_ < n) {
            if (pathx_) {delete[] pathx_;}
            if (pathy_) {delete[] pathy_;}
            pathx_ = new float[n];
            pathy_ = new float[n];
            npathbuf_ = n;
        }
        int * st;
        st = goal_;
        int stc = st[1] * nx_ + st[0];
        npath_ = 0;

        for(int i = 0; i < n; i++)
        {
            pathx_[npath_] = stc % nx_;
            pathy_[npath_] = stc / nx_;
            npath_++;
            
            if (potarr_[stc] < COST_NEUTRAL) {
                return npath_;  // done!
            }

            int stcnx = stc + nx_;
            int stcpx = stc - nx_;

            int minc = stc;
            int curp = potarr_[stc];
            int minp = curp;

            if (potarr_[stcpx] < minp) {minp = potarr_[stcpx]; minc = stcpx;}
            if (potarr_[stcnx] < minp) {minp = potarr_[stcnx]; minc = stcnx;}
            int st;
            st = stc - 1;
            if (potarr_[st] < minp) {minp = potarr_[st]; minc = st;}
            st = stc + 1;
            if (potarr_[st] < minp) {minp = potarr_[st]; minc = st;}

            stc = minc;

            if(minp >= curp)
            {
                RCLCPP_DEBUG(rclcpp::get_logger("rclcpp"), "[PathCalc] Zero gradient");
                return 0;
            }
        }

        RCLCPP_DEBUG(rclcpp::get_logger("rclcpp"), "[PathCalc] No path found, path too long");
        return 0;
    }



}