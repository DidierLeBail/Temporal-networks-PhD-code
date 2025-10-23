
class Solution:
    def getMinFinal(self, test, prev_min):
        if test <= 0:
            res_min = prev_min - test + 1
            res_final = 1
        else:
            res_min = prev_min
            res_final = test
        return res_min, res_final
    
    def init_loc(self):
        """fill the initial location"""
        if self.dungeon[0][0] <= 0:
            self.init_health[(0, 0)] = {1: 1 - self.dungeon[0][0]}
        else:
            self.init_health[(0, 0)] = {1 + self.dungeon[0][0]: 1}
    
    def first_row(self):
        """Fill the first row.
        
        Since there is a single path from (0, 0) to (0, j), we only need to store a single key in `init_health[(0, j)]`
        """
        for j in range(1, self.n):
            prev_final = list(self.init_health[(0, j - 1)])[0]
            prev_init = self.init_health[(0, j - 1)][prev_final]

            # test whether the initial health to arrive at the previous location is enough to overcome the current room
            test = prev_final + self.dungeon[0][j]
            new_init, new_final = self.getMinFinal(test, prev_init)

            self.init_health[(0, j)] = {new_final: new_init}
    
    def first_col(self):
        """Fill the first column.
        
        Since there is a single path from (0, 0) to (i, 0), we only need to store a single key in `init_health[(i, 0)]`"""
        for i in range(1, self.m):
            prev_final = list(self.init_health[(i - 1, 0)])[0]
            prev_init = self.init_health[(i - 1, 0)][prev_final]

            # test whether the initial health to arrive at the previous location is enough to overcome the current room
            test = prev_final + self.dungeon[i][0]
            new_init, new_final = self.getMinFinal(test, prev_init)

            self.init_health[(i, 0)] = {new_final: new_init}
   
    def _garbage(self):
        test1 = final_health[i - 1][j] + dungeon[i][j]
        new_min1, new_final1 = self.getMinFinal(test1, min_health[i - 1][j])

        test2 = final_health[i][j - 1] + dungeon[i][j]
        new_min2, new_final2 = self.getMinFinal(test2, min_health[i][j - 1])

        if new_final1 < new_final2:
            min_health[i][j] = new_min2
            final_health[i][j] = new_final2
        else:
            min_health[i][j] = new_min1
            final_health[i][j] = new_final1
        print(i, j)
        print(min_health[i][j])
        print(final_health[i][j])
        print()

        print("first column")
        for i in range(1, self.m):
            print("final pos", (i, 0))
            print(self.init_health[(i, 0)])
            print()
        
        print("first row")
        for j in range(1, self.n):
            print("final pos", (0, j))
            print(self.init_health[(0, j)])
            print()

    def previous_paths(self, i, j):
        """Iterate over all memorized paths arriving at (i, j).

        For each path, return the pair (test_final, prev_init).
        """
        for pos in [(i - 1, j), (i, j - 1)]:
            for final_health, init_health in self.init_health[pos].items():
                yield final_health + self.dungeon[i][j], init_health

    def calculateMinimumHP(self, dungeon: List[List[int]]) -> int:
        """Calculate iteratively (start with first row and column, then browse
        each location from left to right, top to bottom), for each location,
        the following dictionary:
        - keys are values for the final health (health the knight has after arriving at the location)
        - dic[key] is the minimum initial health of the knight, required so that it makes it to the current location
        with key as final amount of health
        Each pair (key, value) corresponds to a given path the knight has taken from the initial to the current location.
        The maximum number of keys that must be stored is the maximum number of values for the final health of the knight.
        This number is at most of O(n * m).
        """
        # stores the dungeon and its size
        self.dungeon = dungeon
        self.m = len(dungeon)
        self.n = len(dungeon[0])
        
        # init_health[(i, j)][final_health] = minimum initial health required for the knight to arrive at (i, j) with `final_health` as health.
        self.init_health = {(i, j): {} for i in range(self.m) for j in range(self.n)}

        # fill init_health[(0, 0)] (handle the initial location)
        self.init_loc()
        
        # handle the first row
        self.first_row()
        
        # handle the first column
        self.first_col()
        
        # fill the center
        for i in range(1, self.m):
            for j in range(1, self.n):
                # for each possible level of final health, we want to determine the minimum initial health
                # required to arrive at (i, j) with that level of final health
                for args in self.previous_paths(i, j):
                    new_init, new_final = self.getMinFinal(*args)
                    try:
                        self.init_health[(i, j)][new_final] = min(self.init_health[(i, j)][new_final], new_init)
                    except KeyError:
                        self.init_health[(i, j)][new_final] = new_init

            # release memory by removing the previous row
            for j in range(self.n):
                del self.init_health[(i - 1, j)]

        return min(self.init_health[(self.m - 1, self.n - 1)].values())
