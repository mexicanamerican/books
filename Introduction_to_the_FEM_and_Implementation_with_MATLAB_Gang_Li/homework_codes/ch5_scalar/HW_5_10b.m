clear all;

fprintf('Exact result= %f \n', 4/15);
for i=1:3
  [gauss_points, gauss_weights]=GetQuadGauss(i,i);
  I=0;
  for k=1:i*i
    I=I+ gauss_points(k,1)^4*gauss_points(k,2)^2*gauss_weights(k,1);
  end
  fprintf('%d x %d Gauss quadrature I = %f \n',i,i, I);
 end