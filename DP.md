# DP杂谈

#### 线性DP

线性DP指的是具有线性阶段划分的动态规划。

##### [P1868 饥饿的奶牛](https://www.luogu.com.cn/problem/P1868)

线性DP板子题。

用 $f_i$ 表示前 $i$ 个区间能吃到最多的牧草，若有区间 $(i,j)$，则易得到转移：
$$
f_j=\max_{i}(f_{i-1}+(j-i+1))
$$
 用 vector 预处理出 $j$ 对应的 $i$ 即可。

```c++
#include<bits/stdc++.h>
using namespace std;
inline int read(){
	int s=0,w=1;char ch=getchar();
	while(ch<'0'||ch>'9'){if(ch=='-')w=-1;ch=getchar();}
	while(ch>='0'&&ch<='9')s=(s<<1)+(s<<3)+ch-'0',ch=getchar();
	return s*w;
}
const int N=3e6+5;
int n,f[N],maxn;
vector <int> vec[N];
int main(){
	n=read();
	for(int i=1;i<=n;i++){
		int x=read(),y=read();
		vec[y].push_back(x);
		maxn=max(maxn,y);
	}
	for(int i=1;i<=maxn;i++){
		f[i]=f[i-1];
		for(int j=0;j<vec[i].size();j++){
			int x=vec[i][j];
			f[i]=max(f[i],f[x-1]+i-x+1);
		}
	}
	printf("%d",f[maxn]);
	return 0;
} 
```

##### [P4158 [SCOI2009]粉刷匠](https://www.luogu.com.cn/problem/P4158)

需要 DP 两次的题。

令 $f_{i,j}$ 表示前 $i$ 块版刷 $j$ 能正确刷的最多格子，$g_{i,j,k}$ 表示第 $i$ 块板前 $k$ 格粉刷 $j$ 次能最多粉刷的格子数。

对于 $f$ ，显然有：
$$
f_{i,j}=\max_k(f_{i-1,j-k}+g_{i,k,m})
$$
然后考虑求 $g$，枚举一个 $q$，比较 $[q+1,k]$ 刷哪种颜色更划算再转移。
$$
g_{i,j,k}=\max_q(g_{i,j-1,q}+\max(sum_{i,k}-sum_{i,q},k-q-(sum_{i,k}-sum_{i,q})))
$$
 （$sum_{i,j}$ 表示 $i$ 行前 $j$ 格的蓝色/红色个数）

```c++
#include<bits/stdc++.h>
using namespace std;
const int Maxn=55;
char s[Maxn];
int n,m,t,f1[Maxn][Maxn*Maxn][Maxn],f2[Maxn][Maxn*Maxn],sum[Maxn][Maxn*Maxn],ans;
int main(){
	scanf("%d%d%d",&n,&m,&t);
	for(int i=1;i<=n;i++){
		scanf("%s",s);
		for(int j=1;j<=m;j++)
			sum[i][j]=sum[i][j-1]+s[j-1]-'0';
	}
	for(int i=1;i<=n;i++){
		for(int j=1;j<=m;j++){
			for(int k=1;k<=m;k++){
				for(int q=j-1;q<=k-1;q++)
					f1[i][j][k]=max(f1[i][j][k],f1[i][j-1][q]+max(sum[i][k]-sum[i][q],k-q-sum[i][k]+sum[i][q]));
			}	
		}	
	}
	for(int i=1;i<=n;i++){
		for(int j=1;j<=t;j++){
			for(int k=0;k<=min(j,m);k++)
				f2[i][j]=max(f2[i][j],f2[i-1][j-k]+f1[i][k][m]); 
		}	
	}
	for(int i=1;i<=t;i++)ans=max(ans,f2[n][i]);
	printf("%d",ans);
	return 0;
}
```

##### [P3558 [POI2013]BAJ-Bytecomputer](https://www.luogu.com.cn/problem/P3558)

暴力分类讨论题。

用 $f_{i,j}$ 表示前 $i$ 个已经单调不降时，第 $i$ 个数为 $j-1$ 需要的最少操作数。

当  $a_i=-1$ 时：

如果 $a_i$ 将会改为 $-1$，所以不需要操作，并且第 $i-1$ 位只能是 $-1$：
$$
f_{i,0}=f_{i-1,0}
$$
如果 $a_i$ 将会改为 $0$，由于单调不降性，第 $i-1$ 位必须初始为 $1$ 才能转移，显然此时需要一次操作：
$$
f_{i,1}=\min(f_{i-1,0},f_{i-1,1})+1(a_{i-1}=1)
$$


$$
f_{i,1}=inf(a_{i-1}\ne 1)
$$

如果 $a_i$ 将会改为 $1$，如果 $a_{i-1}=1$，则可以在任意时刻转移，反之只能在前一位修改为 $1$ 时才能修改，显然需要两次操作：
$$
f_{i,2}=\min(f_{i-1,0},f_{i-1,1},f_{i-1,2}+2)(a_{i-1}=1)
$$



$$
f_{i,2}=f_{i-1,2}+2
$$

同理可以求出 $a_i=1$ 时和 $a_i=2$ 时的方程。

```c++
#include<bits/stdc++.h>
using namespace std;
const int Inf=(1<<30),Maxn=1e6+5;
inline int read(){
	int s=0,w=1;char ch=getchar();
	while(ch<'0'||ch>'9'){if(ch=='-')w=-1;ch=getchar();}
	while(ch>='0'&&ch<='9')s=(s<<1)+(s<<3)+ch-'0',ch=getchar(); 
	return s*w;
}
int n,a[Maxn],f[Maxn][3];
int main(){
	n=read();
	for(int i=1;i<=n;i++)a[i]=read();
	f[1][0]=f[1][1]=f[1][2]=Inf;
	f[1][a[1]+1]=0;
	for(int i=2;i<=n;i++){
		if(a[i]==-1){
			f[i][0]=f[i-1][0];
			f[i][1]=a[i-1]==1?min(f[i-1][0],f[i-1][1])+1:Inf;
			f[i][2]=a[i-1]==1?min(f[i-1][0],min(f[i-1][1],f[i-1][2]))+2:f[i-1][2]+2; 
		}
		if(a[i]==0){
			f[i][0]=f[i-1][0]+1;
			f[i][1]=min(f[i-1][1],f[i-1][0]);
			f[i][2]=a[i-1]==1?min(f[i-1][0],min(f[i-1][1],f[i-1][2]))+1:f[i-1][2]+1;
		}
		if(a[i]==1){
			f[i][0]=f[i-1][0]+2;
			f[i][1]=a[i-1]==-1?min(f[i-1][0],f[i-1][1])+1:f[i-1][0]+1;
			f[i][2]=min(f[i-1][0],min(f[i-1][1],f[i-1][2]));
		}
	}
	int ans=min(f[n][0],min(f[n][1],f[n][2]));
	if(ans>=Inf)printf("BRAK");
	else printf("%d",ans);
	return 0;
}
```

推荐题目：[P3336 [ZJOI2013]话旧](https://www.luogu.com.cn/problem/P3336)（更有思维难度的分类讨论，基本的组合知识，细节处理）

蒟蒻的题解：[Link](https://www.luogu.com.cn/blog/zrqr-12314/solution-p3336)



##### [P2501 [HAOI2006]数字序列](https://www.luogu.com.cn/problem/P2501)

这题其实和 DP 关系不大 ）

先考虑第一问。
显然，要求 $a_i-a_{i-1}>=1$​，即 $a_i-a_j>=i-j$​，移项得 $a_i-i>=a_j-j$​（$i>j$​）。
所以构造一个数列 $b$​ 使 $b_i=a_i-i$​，那么 $b$​ 的最长不降子序列以外的数要改。
然后考虑第二问。
对于其中已经单调不降的最长子序列，令子序列中两个相邻的项对应原序列的下标为 $i$​，$j$​（$i<j$​），那么原序列 $i$​，$j$​ 之间的点就需要修改。
对于任意一种合法的方案，我们把一些最长连续并且值相同的项称为台阶（台阶的长度可以为1）。
对于每个台阶，有两个量，上升值表示台阶中值增加的项的个数，下降值表示台阶中值减少的项的个数。
如果下降值小于上升值，则将台阶中所有项的值变为左边台阶的值就可以是变化的值减少。
反之，将台阶中所有的值变为右边台阶的值。
所以对于 $i$​，$j$​ 之间的数修改的最优方案（之一），一定有一个 $k$​（$i\leq k<j$​），使得 $i$​ 到 $k$​ 内的值相等，$k+1$​ 到 $j$​ 的值相等。
对于每个 $i$​，$j$​ 枚举 $k$​ 即可，用 DP 统计答案。

```c++
#include<bits/stdc++.h>
#define int long long
using namespace std;
const int Maxn=1e5+5;
inline int read(){
	int s=0,w=1;char ch=getchar();
	while(ch<'0'||ch>'9'){if(ch=='-')w=-1;ch=getchar();}
	while(ch>='0'&&ch<='9')s=(s<<1)+(s<<3)+ch-'0',ch=getchar();
	return s*w;
}
int n,b[Maxn],f[Maxn],len,c[Maxn],dp[Maxn],suml[Maxn],sumr[Maxn];
vector <int> vec[Maxn];
signed main(){
	n=read();
	for(int i=1;i<=n;i++)b[i]=read()-i;
	b[n+1]=1e9;
	for(int i=1;i<=n+1;i++){
		int l=0,r=len;
		while(l<r){
			int mid=(l+r+1)>>1;
			if(f[mid]<=b[i])l=mid;
			else r=mid-1;
		}
		if(l==len)++len;
		f[l+1]=b[i];
		c[i]=l+1;
		vec[c[i]].push_back(i);
	}
	vec[0].push_back(0);
	b[0]=-1e9;
	memset(f,20,sizeof(f));f[0]=0;
	for(int i=1;i<=n+1;i++){
		for(int j=0;j<vec[c[i]-1].size();j++){
			int x=vec[c[i]-1][j];
			if(x>i||b[x]>b[i])continue;
			suml[x]=sumr[i-1]=0;
			for(int k=x+1;k<=i-1;k++)suml[k]=suml[k-1]+abs(b[x]-b[k]);
			for(int k=i-2;k>=x;k--)sumr[k]=sumr[k+1]+abs(b[i]-b[k+1]);
			for(int k=x;k<=i-1;k++)f[i]=min(f[i],f[x]+suml[k]+sumr[k]);
		} e
	}
	printf("%lld\n%lld",n-len+1,f[n+1]);
	return 0;
}
```



#### 区间DP

区间DP是以区间为阶段的动态规划。

##### [P4170 [CQOI2007]涂色](https://www.luogu.com.cn/problem/P4170)

区间DP板子题。

用 $f_{i,j}$ 表示 $[i,j]$ 子串内最少需要的染色次数，初始 $f_{i,i}=1$。

当 $s_i=s_j$ 时，涂 $s_i$ 或 $s_j$ 时把另一个一起涂了，既：
$$
f_{i,j}=\min(f_{i,j-1},f_{i+1,j})(s_i=s_j)
$$
当 $s_i \ne s_j$ 时，不会一次涂 $[i,j]$，考虑将 $[i,j]$ 分成两段计算，枚举断点 $k$：
$$
f[i][j]=\min_k(f[i][k],f[k+1][j])(s[i]\ne s[j],i\leq k\leq j)
$$

```c++
#include<bits/stdc++.h>
using namespace std;
char a[55];
int f[55][55];
int main(){
	scanf("%s",a+1);
	int n=strlen(a+1);
	memset(f,0x3f,sizeof(f));
	for(int i=1;i<=n;i++)f[i][i]=1;
	for(int len=1;len<n;len++){
		for(int l=1;l<=n-len;l++){
			int r=l+len;
			if(a[l]==a[r]){
				f[l][r]=min(f[l-1][r],f[l][r-1]);
				continue;
			}
			for(int k=l;k<r;k++)f[l][r]=min(f[l][r],f[l][k]+f[k+1][r]);
		}
	}
	printf("%d",f[1][n]);
	return 0;
}
```

类似的水题还有：[P3146 [USACO16OPEN]248 G](https://www.luogu.com.cn/problem/P3146)	[P1063 [NOIP2006 提高组] 能量项链](https://www.luogu.com.cn/problem/P1063)







##### [P2466 [SDOI2008] Sue 的小球](https://www.luogu.com.cn/problem/P2466)

一道提前计算贡献的好题。

在之前的动态规划中，我们总是在转移状态时计算贡献，但这道题中当前的决策对以后的状态产生影响。

所以我们考虑在决策时提前算出对未来的贡献，将所有点按坐标排序，这样题目转化为从起点开始一直向任意方向走向最近的点，令到达 $i$ 点的时间为 $t_i$，那么这个点的贡献就为 $\sum y_i-t_i*v_i$，这样这个题就像区间 DP 啦！

用 $f_{i,j,0/1}$ 表示已经射中 $[i,j]$ 内的所有点，停留在 $i/j$  上的最大得分。

由于转移时，未来的得分会累积减少 $t_i*v_i$ ，所以将这部分预处理出来后提前计算：
$$
f_{i,j,0}=y_i+ \max_j(f_{i+1,j,0}−(x_{i+1}−x_i)∗w_{i+1,j},f_{i+1,j,1}−(x_j−x_i)∗w_{i+1,j})
$$

$$
f_{i,j,1}=y_j+\max_i(f_{i,j-1,1}−(x_j−x_{j-1})∗w_{i,j-1},f_{i,j-1,0}−(x_j−x_i)∗w_{i,j-1})
$$

$$
w_{i,j}=\sum_{k_1=1}^nvk_1-\sum_{k_2=i}^jvk_2
$$

```c++
#include<bits/stdc++.h>
#define ll long long
using namespace std;
const int Maxn=1005;
int n;
ll sx,f[2][Maxn][Maxn],sum[Maxn];
struct egg{
	ll x,y,v;
	bool operator <(const egg &X)const{return x<X.x;}
}a[Maxn];
inline ll calc(int i,int j){return sum[n+1]+sum[i-1]-sum[j];}
signed main(){
	scanf("%d%lld",&n,&sx);
	a[1].x=sx,a[1].y=0,a[1].v=0;
	for(int i=2;i<=n+1;i++)scanf("%lld",&a[i].x);
	for(int i=2;i<=n+1;i++)scanf("%lld",&a[i].y);
	for(int i=2;i<=n+1;i++)scanf("%lld",&a[i].v);
	sort(a+1,a+n+2);
	int st;
	for(int i=1;i<=n+1;i++){
		sum[i]=sum[i-1]+a[i].v;
		if(a[i].y==0&&a[i].x==sx)st=i;
	}
	for(int i=0;i<=n+1;i++){
		for(int j=0;j<=n+1;j++)
			f[1][i][j]=f[0][i][j]=-1e18;
	}
	f[0][st][st]=f[1][st][st]=0;
	for(int len=1;len<=n+1;len++){
		for(int l=1;l+len<=n+1;l++){
			int r=l+len;
			f[0][l][r]=a[l].y+max(f[0][l+1][r]-(a[l+1].x-a[l].x)*calc(l+1,r),f[1][l+1][r]-(a[r].x-a[l].x)*calc(l+1,r));
			f[1][l][r]=a[r].y+max(f[0][l][r-1]-(a[r].x-a[l].x)*calc(l,r-1),f[1][l][r-1]-(a[r].x-a[r-1].x)*calc(l,r-1));
		}
	}
	printf("%.3lf",max(f[0][1][n+1],f[1][1][n+1])/1000.0);
	return 0;
}
```



#### 树形DP

树形DP 是在树上的动态规划，一般采用递归（遍历整颗树）实现。

##### [P2014 [CTSC1997]选课](https://www.luogu.com.cn/problem/P2014)

树形 DP 板子题。

由题意得，建出的图是森林，所以建一个根节点，就转化成了一棵树。

易想到用 $f_{x,i,j}$ 表示第 $x$ 个节点的子树中，前 $i$ 颗子树选出 $j$ 门课的最大得分，类比背包，若 $y$ 是 $x$  的儿子则有：
$$
f_{x,i,j}=\max_k(f_{x,i-1,j-k}+f_{x,q,k})
$$
$q$ 表示 $y$ 的儿子数量。

类比背包，可以倒序枚举 $k$ ，这样就可以优化掉 $i$ 这维。

```c++
#include<bits/stdc++.h>
using namespace std;
vector <int> son[305];
int n,m,f[305][305];
inline int read(){
	int s=0,w=1;
	char ch=getchar();
	while(ch<'0'||ch>'9'){if(ch=='-')w=-1;ch=getchar();}
    while(ch>='0'&&ch<='9')s=s*10+ch-'0',ch=getchar();
    return s*w;
}
void dp(int x){
	for(int i=0;i<son[x].size();i++){
		int y=son[x][i];
		dp(y);
		for(int t=m+1;t>=1;t--){
			for(int j=0;j<t;j++)
				f[x][t]=max(f[x][t],f[x][t-j]+f[y][j]);
		}
	}
}
int main(){
	n=read(),m=read();
	for(int i=1;i<=n;i++){
		son[read()].push_back(i);
		f[i][1]=read();
	}
	dp(0);
	printf("%d",f[0][m+1]);
	return 0;
}
```

##### [P3047 [USACO12FEB]Nearby Cows G](https://www.luogu.com.cn/problem/P3047)

树上 DP 两次的题目。

用 $f_{i,j}$ 表示与第 $i$ 个节点相距 $j$ 的节点个数，考虑分两段来求。

先求出节点在 $i$ 的子树内的情况，若 $y$ 是 $i$ 的儿子，易得：
$$
f_{i,j}=\sum_kf_{k,j-1}
$$
再考虑用 $i$ 来更新 $k$，因为统计距 $i$ 为 $j$  的点时一定包含距 $k$ 为 $j-1$  的点，而这些点已经统计过了，容斥一下易得：
$$
f_{k,j}=\sum_i{f_{i,j-1}-f_{k,j-2}}
$$
这里需要注意循环顺序。

```c++
#include<bits/stdc++.h>
using namespace std;
inline int read(){
	int s=0,w=1;char ch=getchar();
	while(ch<'0'||ch>'9'){if(ch=='-')w=-1;ch=getchar();}
	while(ch>='0'&&ch<='9')s=(s<<1)+(s<<3)+ch-'0',ch=getchar();
	return s*w;
}
const int Maxn=200005;
int n,k,head[Maxn],nxt[Maxn],ver[Maxn],tot,f[Maxn][25];
inline void add(int x,int y){ver[++tot]=y,nxt[tot]=head[x],head[x]=tot;}
void dfs1(int x,int fa){
	for(int i=head[x];i;i=nxt[i]){
		int y=ver[i];
		if(y==fa)continue;
		dfs1(y,x);
		for(int j=1;j<=k;j++)f[x][j]+=f[y][j-1];
	}
}
void dfs2(int x,int fa){
	for(int i=head[x];i;i=nxt[i]){
		int y=ver[i];
		if(y==fa)continue;
		for(int j=k;j>=2;j--)f[y][j]-=f[y][j-2];
		for(int j=1;j<=k;j++)f[y][j]+=f[x][j-1];
		dfs2(y,x);
	}
}
int main(){
	n=read(),k=read();
	for(int i=1;i<n;i++){
		int x=read(),y=read();
		add(x,y),add(y,x);
	}
	for(int i=1;i<=n;i++)f[i][0]=read();
	dfs1(1,0);dfs2(1,0);
	for(int i=1;i<=n;i++){
		int sum=0;
		for(int j=0;j<=k;j++)sum+=f[i][j];
		printf("%d\n",sum);
	}
	return 0;
}
```

##### [P2607 [ZJOI2008]骑士](https://www.luogu.com.cn/problem/P2607)

按题意建图得到这是一个基环树森林，考虑强行删去每个环上的一条边使其变为森林。

分别强制删去边的两个端点不选，然后跑树形DP。

用 $f_{x,0/1}$ 表示以 $x$ 为根的子树选/不选 $x$ 的最大战斗力，若 $y$ 是 $x$ 的儿子，易得：
$$
f_{x,0}=\sum \max(f_{y,0},f_{y,1})
$$

$$
f_{x,1}=\sum f_{y,0}
$$

```c++
#include<bits/stdc++.h>
#define int long long
using namespace std;
inline int read(){
	int s=0,w=1;char ch=getchar();
	while(ch<'0'||ch>'9'){if(ch=='-')w=-1;ch=getchar();}
	while(ch>='0'&&ch<='9')s=(s<<1)+(s<<3)+ch-'0',ch=getchar();
	return s*w;
}
const int N=1e6+5;
int n,a[N],head[N],ver[N<<1],nxt[N<<1],tot,ans;
inline void add(int x,int y){ver[++tot]=y,nxt[tot]=head[x],head[x]=tot;}
bool vis[N];int e,x1,x2,f[N][2];
void find(int x,int fa){
	vis[x]=true;
	for(int i=head[x];i;i=nxt[i]){
		int y=ver[i];
		if(y==fa)continue;
		if(vis[y]){
			e=i,x1=x,x2=y;
			continue;
		}
		find(y,x);
	}
}
void dfs(int x,int fa){
	f[x][0]=0,f[x][1]=a[x];
	for(int i=head[x];i;i=nxt[i]){
		int y=ver[i];
		if(y==fa||i==e||(i^1)==e)continue;
		dfs(y,x);
		f[x][1]+=f[y][0];
		f[x][0]+=max(f[y][1],f[y][0]);
	}
}
signed main(){
	tot=1;
	n=read();
	for(int i=1;i<=n;i++){
		a[i]=read();int x=read();
		add(i,x),add(x,i);
	}
	for(int i=1;i<=n;i++){
		if(vis[i])continue;
		find(i,0);
		dfs(x1,0);
		int dat=f[x1][0];
		dfs(x2,0);
		dat=max(dat,f[x2][0]);
		ans+=dat;
	}
	printf("%lld",ans);
	return 0;
}
```

##### [P3177 [HAOI2015]树上染色](https://www.luogu.com.cn/problem/P3177)

将两个同色点之间的贡献转化为两点之间边的贡献，方便计算。

用 $f_{x,k}$ 表示在 $x$ 的子树中选出 $k$ 个黑色点时，子树中所有边的最大贡献。

若 $y$ 是 $x$ 的儿子转移时只用知道边 $(x,y)$ 的贡献即可，而且这个贡献可以根据 $k$ $O(1)$ 求出，易得：
$$
f_{x,k}=\max_j{f_{y,k-j}+(m-k)*k+(n-m+k-size_x)*(size_x-k)}
$$
$size_x$ 表示 $x$ 的子树大小。

```c++
#include<bits/stdc++.h>
#define int long long
using namespace std;
inline int read(){
	int s=0,w=1;char ch=getchar();
	while(ch<'0'||ch>'9'){if(ch=='-')w=-1;ch=getchar();}
	while(ch>='0'&&ch<='9')s=(s<<1)+(s<<3)+ch-'0',ch=getchar();
	return s*w;
}
const int M=4005;
int n,m,head[M],nxt[M],edge[M],ver[M],tot;
inline void add(int x,int y,int z){ver[++tot]=y,edge[tot]=z,nxt[tot]=head[x],head[x]=tot;}
int f[M][M],siz[M];
void dfs(int x,int fa){
	f[x][0]=f[x][1]=0;siz[x]=1;
	for(int i=head[x];i;i=nxt[i]){
		int y=ver[i],z=edge[i];
		if(y==fa)continue;
		dfs(y,x);
		siz[x]+=siz[y];
		for(int j=min(m,siz[x]);j>=0;j--){
			if(f[x][j]!=-1)f[x][j]+=f[y][0]+siz[y]*(n-m-siz[y])*z;
			for(int k=min(j,siz[y]);k>=1;k--){
				if(f[x][j-k]==-1)continue;
				int w=(k*(m-k)+(siz[y]-k)*(n-m-siz[y]+k))*z;   
				f[x][j]=max(f[x][j],f[x][j-k]+f[y][k]+w);
			}
		}
	} 
}
signed main(){
	n=read(),m=read();
	if(n-m<m)m=n-m;
	for(int i=1;i<n;i++){
		int x=read(),y=read(),z=read();
		add(x,y,z),add(y,x,z);
	}
	memset(f,-1,sizeof(f));
	dfs(1,0);
	printf("%lld",f[1][m]);
	return 0;
}
```

树形DP分类讨论题：[P4516 [JSOI2018]潜入行动](https://www.luogu.com.cn/problem/P4516)（不需要任何技巧，暴力分类即可）



