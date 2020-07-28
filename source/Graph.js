

class Graph {

    constructor(vertices) {
        this.vertices = vertices;
        this.edge = new Array(vertices+1);
        var i;
        for (i = 0; i <= vertices; i++) {
            this.edge[i] = [];
        }
        //console.log(this.edge);
    }

    addEdge(a, b) {
        this.edge[a].push(b);
    }

    storeDFS(node, adj, dp, visited, last_node, order) {
        visited[node] = true;
        order.push(node);

        var i;
        for (i = 0; i < adj[node].length; i++) {
            if (!visited[adj[node][i]]) {
                last_node[node] = Math.max(last_node[node], adj[node][i]);
                this.storeDFS(adj[node][i], adj, dp, visited, last_node, order);
            }
            dp[node] = Math.max(dp[node], 1 + dp[adj[node][i]])
        }
        return last_node;
    }

    findLongestPathSrc(n, src) {

        var adj = this.edge;

        var dp = new Array(n+1).fill(0);
        var visited = new Array(n+1).fill(false);
        var last_node = new Array(n+1).fill(-1);
        var order = [];

        var i;
        for (i=0;i<=n;i++) {
            last_node[i] = i;
        }

        this.storeDFS(src, adj, dp, visited, last_node, order);
        var ans = 0;
        for (i = 1; i <= n; i++) {
            ans = Math.max(ans, dp[i]);
        }
        // first one is total traversal length, second one is last node
        return [last_node[src], order];
    }


}


// Now have to actually draw graph from visited nodes Order



var n = 5;
var graph = new Graph(n);
    // Example-1
graph.addEdge( 1, 2);
graph.addEdge( 1, 3);
graph.addEdge( 3, 2);
graph.addEdge( 2, 4);
graph.addEdge( 3, 4);
graph.addEdge(4,5);
console.log(graph.edge);
console.log("Last node visited: ", graph.findLongestPathSrc(5, 2));


module.exports = {
    Graph: Graph
}