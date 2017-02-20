// Andrew Long
// Updated: Feb 13, 2017
// Doubly linked list acting on a 2D grid (1D representation)
// could probably extend to ND grids but for now this is easiest
// Requirements:
//      - Lookup of the top K values in the matrix
//      - Zero a column or row of the matrix
//      - Undo the last set of changes (lots of pushing and popping onto the stack, don't want pass by value)

// The GridDLL has the following properties:
// stores values in N1xN2 grid as a 1D array, linked list ordered by sorted values
// lazy deletion of matrix indices (invalidates the node and refreshes pointers around it)
// fast reinclusion of deleted nodes via a deleted item stack
// O(K) selection of topK values
// O(2*N2) removal of overlapping points
// O(N1+N2-2*i-1) to restore overlapped points from deletion stack

// Alternatives:
// (1) Vector of sorted pairs of <value, location(i,j)>
// initializiation is the same (requires sort)
// Lookup is O(K)
// Removal of overlap points is O(N1*N2)
// Undo is better managed via pass by value (requires O(N1*N2) copy)
//      - alternative is challenging, would need more bookkeeping and would be slower
//      - inserting into sorted list using the refresh_stack and a secondary copy of the matrix
//
// (2) Raw 1D form of a matrix
//  Initialization is O(1)
//  Lookup is O(N*k) (scan for top value through iteration, some kind of sorted buffer could also work)
//  Removal of overlap points is O(2*N2) (just zero out the indices in the matrix)
//  Undo could be done via a deletion stack O(N1+N2-2*i+1)
//      - needs a secondary copy of the matrix

#ifndef __GRID_DLL__
#define __GRID_DLL__

#include <vector>
#include <functional>
#include <utility>
#include <armadillo>
#include <cassert>

typedef std::pair<long,long> GridPoint;

class GridDLL
{
public: 
	struct GridNode
	{
		GridNode(long _x=-1, long _y=-1, double _w=0, long p = -1, long n = -1, bool v = false)
			{	set(x,y,w,p,n,v);	}
		void set(long _x, long _y, double _w, long _p, long _n, bool _v = false)
		{
			x = _x;
			y = _y;
			w = _w;
			prev = _p;
			next = _n;
			valid = _v;
		}
		long x,y;
		double w;

		long prev;
		long next;
		bool valid;

		bool operator<(const GridNode& other) { return w < other.w; }
	};
protected:
	std::vector<GridNode> grid;
    std::vector<long> refresh_stack;
	long N1, N2, Nelem;
	long head,tail;
	long count;
private:
public:
	GridDLL()
	{
		reset();
	}
    
	GridDLL(const GridDLL& other)
	{
		grid.assign(other.grid.begin(), other.grid.end());
        
		N1 = other.N1;
		N2 = other.N2;
		Nelem = other.Nelem;
		count = other.count;
		head = other.head;
		tail = other.tail;
        refresh_stack.reserve(Nelem);
	}
    
	~GridDLL()
	{
		reset();
	}

	void reset()
	{
		grid.clear();
        refresh_stack.clear();
		N1 = 0;
		N2 = 0;
		count = 0;
		Nelem = 0;
		head = -1;
		tail = -1;
	}
    
	bool empty()
	{
		return count == 0;
	}
    
    long size()
    {
        return count;
    }
    
	bool consistent()
	{
		long c = 0;
		long n = head;
		while(n != -1)
		{
			++c;
			n = grid[n].next;
		}
		return c == count;
	}
    
	void buildFromMat(const arma::mat& mat)
	{
		N1 = (mat.n_rows);
		N2 = (mat.n_cols);
        Nelem = mat.n_elem;
        count = Nelem;
        
		assert(N1 > 1 && N2 > 1);
		
        arma::uvec sort_idx = arma::sort_index(mat, "descend");
		
        grid.resize(Nelem);
        refresh_stack.reserve(Nelem);
		

		long x,y;
		// construct head element [0]
		x = sort_idx[0]%N1;
		y = sort_idx[0]/N1;

		head = sort_idx[0];
		grid[sort_idx[0]].set(x, y, mat(x,y), -1, sort_idx[1], true);
		
		// construct [1..Nelem-1)
		for(int i = 1; i < Nelem-1; ++i)
		{
			x = sort_idx[i]%N1;
			y = sort_idx[i]/N1;
			
			grid[sort_idx[i]].set(x, y, mat(x,y), sort_idx[i-1], sort_idx[i+1], true);
		}
        
		// construct tail element [Nelem-1]
		x = sort_idx[Nelem-1]%N1;
		y = sort_idx[Nelem-1]/N1;
		tail = sort_idx[Nelem-1];
		grid[sort_idx[Nelem-1]].set(x, y, mat(x,y), sort_idx[Nelem-2],-1, true);
	}
    
    void printList(std::string str = "")
	{
        if(str != "")
            printf("%s\n",str.c_str());
		printf("[Head:%ld]\t[Tail:%ld]\t[Count:%ld]\n",head,tail,count);
		long cur_node = head;
		std::string tabs = "";
		while(cur_node != -1)
		{
			long idx = grid[cur_node].x + (grid[cur_node].y*N1);
			printf("%s[%ld -> %ld -> %ld]\n",tabs.c_str(),grid[cur_node].prev, idx, grid[cur_node].next);
			cur_node = grid[cur_node].next;
			tabs = tabs + "\t";
		}
		printf("\n");
	}
	void print(std::string str = "")
	{
		if(str != "")
			printf("%s\n",str.c_str());

		printf("[Head:(%ld,%ld)]\t[Tail:(%ld,%ld)]\t[Count:%ld]\n",head%N1,head/N1,tail%N1,tail/N1,count);
		long cur_node = head;
		while(cur_node != -1)
		{
			printf("[%ld,%ld (%ld)]:%lf\n",	grid[cur_node].x, grid[cur_node].y, cur_node,grid[cur_node].w);
			cur_node = grid[cur_node].next;
		}
	}
    
	// invalidate the current node and refresh pointers
    // NOTE: assumes the node is valid.
    // OPS: 4x assignment, 2x add/subtract, 3 booleans
	bool deleteNode(long idx)
	{
		if(count == 1)
		{
			head = -1;
			tail = -1;
		}
		else
		{
			if(idx == head)
			{
				head = grid[idx].next;
				grid[head].prev = -1;
			}
			else if(idx == tail)
			{
				tail = grid[idx].prev;
				grid[tail].next = -1;
			}
			else
			{
				long p = grid[idx].prev;
				long n = grid[idx].next;
				grid[p].next = n;
				grid[n].prev = p;
			}
		}
		--count;
		grid[idx].valid = false;
        refresh_stack.push_back(idx);
        return true;
	}
    void checkALL()
    {
        bool b = consistent();
        long c = count;
        printf("CheckALL: consistent:%d, count:%ld\n",b,c);
        printList();
        printf("\nRefreshStack: %ld",refresh_stack[0]);
        for(int i = 1; i < (int)refresh_stack.size(); ++i)   printf(",%ld",refresh_stack[i]);
        printf("\n");
        printf("StackValid: %d",grid[refresh_stack[0]].valid);
        for(int i = 1; i < (int)refresh_stack.size(); ++i)   printf(",%d",grid[refresh_stack[i]].valid);
        printf("\n");
        long d = 0;
        for(int i = 0; i < Nelem; ++i)
        {
            if(grid[i].valid)
                ++d;
        }
        if(d != c)
        {
            printf("Crap");
        }
    }
	void clearOverlaps(long x, long y)
	{
		for(int i = 0; i < N2; ++i)
		{
			long idx_x = i*N1+x;
			if(grid[idx_x].valid)
				deleteNode(idx_x);
		}
		for(int i = 0; i < N1; ++i)
		{
			long idx_y = y*N1+i;
			if(grid[idx_y].valid)
				deleteNode(idx_y);
		}
	}
    void relinkStack(long depth)
    {
        long num_to_pop = N1 + N2 - 1 - 2*depth;
        for(long i = 0; i < num_to_pop; ++i)
        {
            long restore = refresh_stack.back();
            
            // restore the node
            grid[restore].valid = true;
            
            // prev restoration
            if(grid[restore].next != -1)
                grid[grid[restore].next].prev = restore;
            if(head == grid[restore].next)
                head = restore;
            
            // next restoration
            if(grid[restore].prev != -1)
                grid[grid[restore].prev].next = restore;
            if(tail == grid[restore].prev)
                tail = restore;
            
            ++count;
            refresh_stack.pop_back();
        }
    }
    
	long topK(long K, std::vector<GridPoint>& points, long start = 0)
	{
		long nK = std::min(K, count);
		long cur_node = head;
		for(long i = 0; i < nK; ++i)
		{
            points[start+i].first = grid[cur_node].x;
            points[start+i].second = grid[cur_node].y;
			cur_node = grid[cur_node].next;
		}

		return nK;
	}
};

#endif
