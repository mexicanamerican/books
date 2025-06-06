function postproc(dcax,dcbx,dccx,f3,glf,glx,nod,icont,ielem,npe,model,ntype,item,mxelm,mxneq,mxnod,nem,ndf,fid_out)
% c     __________________________________________________________________
% c
% c     the subroutine is called in main to compute the solution and its
% c     derivatives at  five points, including the nodes of the element.
% c     the bending moment (bm) and shear force (vf) are computed as per
% c     the definitions given in fig. 5.2.1 and eq. (5.3.1) of the book.
% c
% c        x........ global (i.e., problem) coordinate
% c        xi ...... local (i.e., element) coordinate
% c        sf....... element interpolation (or shape) functions
% c        gdsf..... first derivative of sf w.r.t. global coordinate
% c        gddsf.... second derivative of sf w.r.t. global coordinate
% c        elu...... element solution vector
% c        u........ interpolated solution
% c        du....... interpolated derivative of the solution
% c        w........ interpolated transverse deflection
% c        s........ interpolated rotation function
% c        ds....... interpolated derivative of the rotation
% c        dw....... interpolated derivative of the transverse deflection
% c        ddw...... interpolated second derivative of trans. deflection
% c        dddw..... interpolated third derivative of trans. deflection
% c     __________________________________________________________________

% implicit real*8 (a-h,o-z)
% dimension dcax(mxelm,2),dcbx(mxelm,2),dccx(mxelm,2)
% dimension f3(mxelm),glf(mxneq),glx(mxnod),nod(mxelm,4)
% dimension xp(9),elx(4),elu(9)
% common/io/in,it
% common/shp/sf(4),gdsf(4),gddsf(4),gj
% common/stf2/a1,a2,a3,a4,a5,ax0,ax1,bx0,bx1,cx0,cx1,ct0,ct1,fx0,fx1,fx2
% data xp/-1.0d0, -0.750d0, -0.50d0, -0.250d0, 0.0d0, 0.250d0,0.50d0,  
%0.750d0, 1.0d0/
% c
global a1 a2 a3 a4 a5 ax0 ax1 bx0 bx1 cx0 cx1 ct0 ct1 fx0 fx1 fx2
xp=[-1.0d0, -0.750d0, -0.50d0, -0.250d0, 0.0d0, 0.250d0,0.50d0,0.750d0, 1.0d0]';
npts=9;
for ne = 1:nem
    if(icont~=1)
        ax0=dcax(ne,1);
        ax1=dcax(ne,2);
        bx0=dcbx(ne,1);
        bx1=dcbx(ne,2);
        cx0=dccx(ne,1);
        cx1=dccx(ne,2);
    end
    l=0;
    for i=1:npe
        ni=nod(ne,i);
        if(icont~=1)
            elx(1)=0.0;
            elx(2)=0.5*glx(ne);
            elx(npe)=glx(ne);
        else
            elx(i)=glx(ni);
        end
        li=(ni-1)*ndf;
        for j=1:ndf
            li=li+1;
            l=l+1;
            elu(l)=glf(li);
        end
    end
    h = elx(npe) - elx(1);
    for ni=1:npts
        xi = xp(ni);
        [gj,sf,dsf,gdsf,gddsf]=shape1d(h,ielem,npe,xi);
        if(model==3)
            w=0.0;
            dw=0.0;
            ddw=0.0;
            xc=elx(1)+0.5*h*(1.0+xi);
            for i=1:4
                w  =w  + sf(i)*elu(i);
                dw =dw + gdsf(i)*elu(i);
                ddw=ddw+ gddsf(i)*elu(i);
            end
            dddw=((elu(1)-elu(3))*2.0/h-(elu(4)+elu(2)))*6.0/(h*h);
            theta=-dw;
            if(ntype==0)
                bm=-(bx0+xc*bx1)*ddw;
                vf=-(bx0+xc*bx1)*dddw - bx1*ddw;
                fprintf(fid_out,'% 4.5e\t% 4.5e\t% 4.5e\t% 4.5e\t% 4.5e\n',xc,w,theta,bm,vf);
                %  write(it,90)xc,w,theta,bm,vf
            else
                anu21=bx0*ax0/ax1;
                di=(bx1^3)/12.0;
                d11=di*ax0/(1.0-bx0*anu21);
                d22=d11*(ax1/ax0);
                d12=bx0*d22;
                bmr=-(d11*ddw*xc+d12*dw);
                bmt=-(d12*ddw*xc+d22*dw);
                if(xc~=0.0)
                    sfv=-d11*(xc*dddw+ddw)+d22*dw/xc;
                    fprintf(fid_out,'\t% 4.5e\t% 4.5e\t% 4.5e\t% 4.5e\t% 4.5e\t% 4.5e\n',xc,w,theta,bmr,bmt,sfv);
                    % write(it,90)xc,w,theta,bmr,bmt,sfv
                else
                    fprintf(fid_out,'\t% 4.5e\t% 4.5e\t% 4.5e\t% 4.5e\t% 4.5e\t\n',xc,w,theta,bmr,bmt);
                    % write(it,90)xc,w,theta,bmr,bmt
                end
            end
        else
            xc=0.0;
            for i=1:npe
                xc=xc+sf(i)*elx(i);
            end
            if(model==1)
                u=0.0;
                du=0.0;
                for i=1:npe
                    u=u+sf(i)*elu(i);
                    du=du+gdsf(i)*elu(i);
                end
                if(ntype==0)
                    sv=(ax0+ax1*xc)*du;
                    fprintf(fid_out,'\t% 4.5e\t% 4.5e\t% 4.5e\t \n',xc,u,sv);
                    % write(it,90)xc,u,sv
                else
                    anu21=bx0*ax0/ax1;
                    if(ntype==1)
                        c11=bx1*ax0/(1.0-bx0*anu21);
                        c22=c11*(ax1/ax0);
                        c12=bx0*c22;
                    else
                        denom=1.0-bx0-anu21;
                        c11=bx1*ax0*(1.0-bx0)/(1.0+bx0)/denom;
                        c22=bx1*ax1*(1.0-anu21)/(1.0+anu21)/denom;
                        c12=bx0*c22;
                    end
                    if(xc~=0.0)
                        sr=c11*du+c12*u/xc;
                        st=c12*du+c22*u/xc;
                        fprintf(fid_out,'\t% 4.5e\t% 4.5e\t% 4.5e\t% 4.5e\t\n',xc,u,sr,st); % write(it,90)xc,u,sr,st
                    else
                        fprintf(fid_out,'\t% 4.5e\t% 4.5e\t% 4.5e\t% \n',xc,u,du); % write(it,90)xc,u,du
                    end
                end
            else
                %c     model.eq.2  calculations
                if(item==0)&&(ntype>1)
                    h=elx(npe)-elx(1);
%                     timstres(ax0,elu,xi,w,dw,psi,dpsi,ne,f3,h,mxelm);
                    gj =  h*0.5;
                    %c     interpolation functions for the lagrange linear element
                    sfl(1) = 0.5*(1.0-xi);
                    sfl(2) = 0.5*(1.0+xi);
                    dsfl(1) = -0.5/gj;
                    dsfl(2) = 0.5/gj;
                    %c     interpolation functions for the lagrange quadratic element
                    sfq(1) = -0.5*xi*(1.0-xi);
                    sfq(2) = 1.0-xi*xi;
                    sfq(3) = 0.5*xi*(1.0+xi);
                    dsfq(1) = -0.5*(1.0-2.0*xi)/gj;
                    dsfq(2) = -2.0*xi/gj;
                    dsfq(3) = 0.5*(1.0+2.0*xi)/gj;
                    %c
                    w3=(3.0*h*f3(ne)/ax0 + 8.0*(elu(1)+elu(3))+ 2.0*(elu(4)-elu(2))*h)/16.0;
                    w =  sfq(1)*elu(1) + sfq(2)*w3 + sfq(3)*elu(3);
                    dw= dsfq(1)*elu(1) +dsfq(2)*w3 +dsfq(3)*elu(3);
                    psi =  sfl(1)*elu(2) + sfl(2)*elu(4);
                    dpsi= dsfl(1)*elu(2) +dsfl(2)*elu(4);
                else
                    w=0.0;
                    dw=0.0;
                    psi = 0.0;
                    dpsi=0.0;
                    for i=1:npe
                        l=2*i-1;
                        w = w + sf(i)*elu(l);
                        dw=dw+gdsf(i)*elu(l);
                        psi = psi + sf(i)*elu(l+1);
                        dpsi=dpsi+gdsf(i)*elu(l+1);
                    end
                    if(ntype==0)||(ntype==2)
                        bm=(bx0+bx1*xc)*dpsi;
                        vf=(ax0+ax1*xc)*(dw+psi);
                        fprintf(fid_out,'\t% 4.5e\t% 4.5e\t% 4.5e\t% 4.5e\t% 4.5e\t\n',xc,w,psi,bm,vf); %write(it,90)xc,w,psi,bm,vf
                    else
                        anu21=bx0*ax0/ax1;
                        di=(bx1^3)/12.0;
                        d11=di*ax0/(1.0-bx0*anu21);
                        d22=d11*(ax1/ax0);
                        d12=bx0*d22;
                        bmr=(d11*dpsi*xc+d12*psi);
                        bmt=(d12*dpsi*xc+d22*psi);
                        sfv=fx2*(dw+psi)*xc;
                        fprintf(fid_out,'\t% 4.5e\t% 4.5e\t% 4.5e\t% 4.5e\t% 4.5e\t% 4.5e\n',xc,w,psi,bmr,bmt,sfv);
                        % write(it,90)xc,w,psi,bmr,bmt,sfv
                    end
                end
            end
        end
    end
end
end


%   647 FORMAT(3X,'x is the global coord. if ICONT=1 and it is the local',
%      *       ' coord. if ICONT=0')
%   650 FORMAT(7X,'  x  ',6X, 'Deflect.',5X,'Rotation',5X,'B. Moment',
%      *      3X,'Shear Force')
%   660 FORMAT(7X,'  x  ',6X, 'Deflect.',5X,'Rotation',4X,'Moment, Mr',
%      *      3X,'Moment, Mt',3X,'Shear Force')
% 
