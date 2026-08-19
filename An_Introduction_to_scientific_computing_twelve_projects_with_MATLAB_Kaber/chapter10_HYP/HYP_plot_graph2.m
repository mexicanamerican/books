%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%    An Introduction to Scientific Computing          %%%%%%%
%%%%%%%    I. Danaila, P. Joly, S. M. Kaber & M. Postel     %%%%%%%
%%%%%%%                 Springer, 2005                      %%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%
%=======================================
% plots on the same graph two curves:
%  u1(xx), u2(xx) which are usually
%  exact solution/ numerical solution
%=======================================

function HYP_plot_graph2(t,xx,u1,u2,txt1,txt2)

yl=['\rho';'U   ';'p   '];yt=['density ';'velocity';'pressure'];
for k=1:3
   fig=figure('Position',0.75*get(0,'Screensize'));   % new figure
   plot(xx,u1(k,:),'r-',xx,u2(k,:),'bo-','LineWidth',2,'MarkerSize',6);
   title([yt(k,:) ' t=' num2str(t) ],'FontSize',24);
   xlabel('x');ylabel(yl(k,:));
   legend(txt1,txt2);   
   print(fig,sprintf('/tmp/out-%d.png',k),'-dpng')

%set(gca,'FontSize',24)
%title([yt(k,:) ' t=' num2str(t) ],'FontSize',24);
%xlabel('x');ylabel(yl(k,:));
%legend(txt1,txt2);
%drawnow;
end
