fd = @(p) -min(min(min(1+p(:,2),1-p(:,2)),1+p(:,1)),1-p(:,1));
fh = @(p) ones(size(p,1),1);
[p,t,e] = distmesh( fd, fh, 0.2, [-1,-1;1,1], [-1,-1;-1,1;1,-1;1,1] );
e