!!Quick routine to wrap the SLALIB moon position function.
!!It is inaccurate but fine for QUIJOTE accuracy. It downsamples
!! the data by 100x!

subroutine refro(zd, hm, tdk, pmb, rh, wl, phi, tlr, eps, ref, len_zd)
  implicit none
  integer, intent(in) :: len_zd
  real*8, intent(in) :: zd(len_zd)
  real*8, intent(out) :: ref(len_zd)
  real*8, intent(in) :: hm ! height
  real*8, intent(in) :: tdk ! ambient temp (K)
  real*8, intent(in) :: pmb ! pressure (mb)
  real*8, intent(in) :: rh ! relative humidity (0-1)
  real*8, intent(in) :: wl ! effective wavelength (um)
  real*8, intent(in) :: tlr ! latitude of observer
  real*8, intent(in) :: phi ! temperature lapse rate (K/m)
  real*8, intent(in) :: eps ! precision required (radian)

  !f2py real*8 zd, hm, tdk, pmb, rh, wl, phi, tlr, eps, ref
  !f2py integer len_zd

  integer :: i

  do i=1, len_zd
     call sla_refro(zd(i), hm, tdk, pmb, rh, wl, phi, tlr, eps, ref(i))
  enddo


end subroutine refro

subroutine rdplan(jd, np, lon, lat, ra, dec, diam, len_bn)
  implicit none
  
  integer, intent(in) :: len_bn
  real*8, intent(in) :: lon
  real*8, intent(in) :: lat
  integer, intent(in) :: np
  real*8, intent(in) :: jd(len_bn)
  real*8, intent(out) :: diam(len_bn)
  real*8, intent(out) :: ra(len_bn)
  real*8, intent(out) :: dec(len_bn)

  !f2py integer len_bn
  !f2py real*8 lon, lat
  !f2py real*8 ra,dec,jd, diam

  !integer :: i

  real*8 :: pi = 3.14159265359

  integer :: step = 100
  integer :: kup,k
  kup = len_bn/step


  do k=1, len_bn
     call sla_rdplan(jd(k),np,lon,lat,ra(k),dec(k),diam(k))
  enddo
  
end subroutine rdplan




subroutine planet(jd, np, dist, len_bn)
  implicit none
  
  integer, intent(in) :: len_bn
  integer, intent(in) :: np
  real*8, intent(in) :: jd(len_bn)
  real*8, intent(out) :: dist(6,len_bn)

  !f2py integer len_bn
  !f2py real*8 jd, dist

  integer :: jstat

  !real*8 :: pi = 3.14159265359

  integer :: step = 100
  integer :: kup,k
  kup = len_bn/step

  !mask = 1.0d0

  do k=1, len_bn
     call sla_planet(jd(k),np,dist(:,k),jstat)
  enddo
  
end subroutine planet

subroutine h2e_full(az, el, mjd, lon, lat,dut, ra, dec, len_bn)
  implicit none
  
  integer, intent(in) :: len_bn
  real*8, intent(in) :: lon
  real*8, intent(in) :: lat
  real*8, intent(in) :: dut
  real*8, intent(in) :: az(len_bn)
  real*8, intent(in) :: el(len_bn)
  real*8, intent(in) :: mjd(len_bn)
  real*8, intent(out) :: ra(len_bn)
  real*8, intent(out) :: dec(len_bn)

  interface
     real*8 FUNCTION sla_gmst(mjddummy)
     real*8 :: mjddummy
     END FUNCTION sla_gmst
  end interface

  interface
     real*8 FUNCTION sla_dranrm(mjddummy)
     real*8 :: mjddummy
     END FUNCTION sla_dranrm
  end interface

  interface
     real*8 FUNCTION sla_dtt(mjddummy)
     real*8 :: mjddummy
     END FUNCTION sla_dtt
  end interface

  !f2py integer len_bn
  !f2py real*8 lon, lat
  !f2py real*8 ra,dec,mjd

  integer :: i

  real*8 :: ra_temp, dec_temp
  real*8 :: pi = 3.14159265359
  real*8 :: zob
  real*8 :: SECPERDAY = 86400.
  real*8 :: djtt
  do i=1, len_bn
     ! call sla_dh2e(az(i), el(i), lat, ra(i), dec(i))
     
     zob = pi / 2D0 - el(i)
     call sla_oap('A',az(i), zob, mjd(i), dut, lon, lat, 0D0, 0D0,0D0,0D0,0D0,0D0,0.55,0D0,ra(i),dec(i))

     djtt = mjd(i) + sla_dtt(mjd(i))/SECPERDAY

     ra_temp = ra(i)
     dec_temp = dec(i)
     ! RA, DEC, DATE, EPOCH
     call sla_amp(ra_temp,dec_temp, djtt, 2000D0,ra(i),dec(i))
     ra(i) = sla_dranrm(ra(i))
  enddo    

end subroutine h2e_full


subroutine e2h_full(ra, dec, mjd, lon, lat,dut, az, el, ha, len_bn)
  implicit none
  
  integer, intent(in) :: len_bn
  real*8, intent(in) :: lon
  real*8, intent(in) :: lat
  real*8, intent(in) :: dut
  real*8, intent(in) :: ra(len_bn)
  real*8, intent(in) :: dec(len_bn)
  real*8, intent(in) :: mjd(len_bn)
  real*8, intent(out) :: az(len_bn)
  real*8, intent(out) :: el(len_bn)
  real*8, intent(out) :: ha(len_bn)

  interface
     real*8 FUNCTION sla_gmst(mjddummy)
     real*8 :: mjddummy
     END FUNCTION sla_gmst
  end interface

  interface
     real*8 FUNCTION sla_dranrm(mjddummy)
     real*8 :: mjddummy
     END FUNCTION sla_dranrm
  end interface

  interface
     real*8 FUNCTION sla_dtt(mjddummy)
     real*8 :: mjddummy
     END FUNCTION sla_dtt
  end interface

  !f2py integer len_bn
  !f2py real*8 lon, lat
  !f2py real*8 ra,dec,mjd

  integer :: i

  real*8 :: ra_temp, dec_temp
  real*8 :: pi = 3.14159265359
  real*8 :: SECPERDAY = 86400.
  real*8 :: djtt
  do i=1, len_bn
     ! call sla_dh2e(az(i), el(i), lat, ra(i), dec(i))
     djtt = mjd(i) + sla_dtt(mjd(i))/SECPERDAY
     ! Convert from Mean to Apparent position
     call sla_map(ra(i),dec(i), 0D0, 0D0, 0D0, 0D0, 2000D0, djtt,ra_temp,dec_temp)
     call sla_aop(ra_temp,dec_temp, mjd(i), dut, lon, lat, 0D0, 0D0, 0D0, 0D0, 0D0, 0D0, 0.55D0, 0D0, & 
          az(i), el(i), ha(i), dec_temp, ra_temp) 
     el(i) = pi / 2D0 - el(i)
  enddo    

end subroutine e2h_full


subroutine h2e(az, el, mjd, lon, lat, ra, dec, len_bn)
  implicit none
  
  integer, intent(in) :: len_bn
  real*8, intent(in) :: lon
  real*8, intent(in) :: lat
  real*8, intent(in) :: az(len_bn)
  real*8, intent(in) :: el(len_bn)
  real*8, intent(in) :: mjd(len_bn)
  real*8, intent(out) :: ra(len_bn)
  real*8, intent(out) :: dec(len_bn)

  interface
     real*8 FUNCTION sla_gmst(mjddummy)
     real*8 :: mjddummy
     END FUNCTION sla_gmst
  end interface


  interface
     real*8 FUNCTION sla_dranrm(mjddummy)
     real*8 :: mjddummy
     END FUNCTION sla_dranrm
  end interface

  interface
     real*8 FUNCTION sla_dtt(mjddummy)
     real*8 :: mjddummy
     END FUNCTION sla_dtt
  end interface

  interface
     real*8 FUNCTION sla_eqeqx(mjddummy)
     real*8 :: mjddummy
     END FUNCTION sla_eqeqx
  end interface


  real*8 :: SECPERDAY = 86400.
  real*8 :: mjd_tt
  real*8 :: eq
  !f2py integer len_bn
  !f2py real*8 lon, lat
  !f2py real*8 ra,dec,mjd

  integer :: i

  real*8 :: gmst

  do i=1, len_bn
     call sla_dh2e(az(i), el(i), lat, ra(i), dec(i))
     gmst = sla_gmst(mjd(i))
     mjd_tt = mjd(i) + sla_dtt(mjd(i))/SECPERDAY
     eq = sla_eqeqx(mjd_tt)
     ra(i) = gmst + lon - ra(i) + eq
     ra(i) = sla_dranrm(ra(i))
  enddo    

  
end subroutine h2e

subroutine e2h(ra, dec, mjd, lon, lat, az, el,lha, len_bn)
  implicit none
  
  integer, intent(in) :: len_bn
  real*8, intent(in) :: lon
  real*8, intent(in) :: lat
  real*8, intent(in) :: ra(len_bn)
  real*8, intent(in) :: dec(len_bn)
  real*8, intent(in) :: mjd(len_bn)
  real*8, intent(out) :: az(len_bn)
  real*8, intent(out) :: el(len_bn)
  real*8, intent(out) :: lha(len_bn)

  interface
     real*8 FUNCTION sla_gmst(mjddummy)
     real*8 :: mjddummy
     END FUNCTION sla_gmst
  end interface


  interface
     real*8 FUNCTION sla_dranrm(mjddummy)
     real*8 :: mjddummy
     END FUNCTION sla_dranrm
  end interface

  interface
     real*8 FUNCTION sla_dtt(mjddummy)
     real*8 :: mjddummy
     END FUNCTION sla_dtt
  end interface

  interface
     real*8 FUNCTION sla_eqeqx(mjddummy)
     real*8 :: mjddummy
     END FUNCTION sla_eqeqx
  end interface


  real*8 :: SECPERDAY = 86400.
  real*8 :: mjd_tt
  real*8 :: eq

  !f2pye2h integer len_bn
  !f2py real*8 lon, lat
  !f2py real*8 az,el,mjd

  integer :: i
  real*8 :: gmst

  do i=1, len_bn
     gmst = sla_gmst(mjd(i))
     mjd_tt = mjd(i) + sla_dtt(mjd(i))/SECPERDAY
     eq = sla_eqeqx(mjd_tt)
     lha(i) = lon + gmst - ra(i) + eq ! CONVERT TO LHA
     call sla_de2h(lha(i), dec(i), lat, az(i), el(i))
  enddo    

  
end subroutine e2h

subroutine precess(ra, dec,mjd, len_bn)
  implicit none
  
  integer, intent(in) :: len_bn
  real*8, intent(in) :: mjd(len_bn)
  real*8, intent(inout) :: ra(len_bn)
  real*8, intent(inout) :: dec(len_bn)

  interface
     real*8 FUNCTION sla_epj(mjddummy)
     real*8 :: mjddummy
     END FUNCTION sla_epj
  end interface
  interface
     real*8 FUNCTION sla_dranrm(mjddummy)
     real*8 :: mjddummy
     END FUNCTION sla_dranrm
  end interface
  interface
     real*8 FUNCTION sla_dtt(mjddummy)
     real*8 :: mjddummy
     END FUNCTION sla_dtt
  end interface

  interface
     real*8 FUNCTION sla_eqeqx(mjddummy)
     real*8 :: mjddummy
     END FUNCTION sla_eqeqx
  end interface


  real*8 :: SECPERDAY = 86400.
  real*8 :: mjd_tt

  !f2py integer len_bn
  !f2py real*8 mjd
  !f2py real*8 ra,dec

  integer :: i
  real*8 :: RNUT(3,3)
  real*8 :: V(3), Vrot(3)

  do i=1, len_bn
     mjd_tt = mjd(i) + sla_dtt(mjd(i))/SECPERDAY
     call sla_prenut(2000D0,mjd_tt,RNUT)
     call sla_dcs2c(ra(i),dec(i),V)
     call sla_dimxv(RNUT,V,Vrot)
     call sla_dcc2s(Vrot,ra(i),dec(i))
     ra(i) = sla_dranrm(ra(i))
  enddo    
  
end subroutine precess

subroutine prenut(ra, dec,mjd, len_bn)
  implicit none
  
  integer, intent(in) :: len_bn
  real*8, intent(in) :: mjd(len_bn)
  real*8, intent(inout) :: ra(len_bn)
  real*8, intent(inout) :: dec(len_bn)

  real*8 :: v1(3)
  real*8 :: v2(3)
  real*8 :: PM(3,3)

  interface
     real*8 FUNCTION sla_epb(mjddummy)
     real*8 :: mjddummy
     END FUNCTION sla_epb
  end interface
  interface
     real*8 FUNCTION sla_dranrm(dummy)
     real*8 :: dummy
     END FUNCTION sla_dranrm
  end interface

  !f2py integer len_bn
  !f2py real*8 mjd
  !f2py real*8 ra,dec

  integer :: i
  real*8 :: epoch

  do i=1, len_bn
     call sla_dcs2c(ra(i),dec(i),v1) ! convert to vectors
     epoch = sla_epb(mjd(i))
     print *, ra(i),dec(i)
     call sla_prenut(epoch,2000D0,PM) ! Nutation and precession matrix
     call sla_dmxv(PM,v1,v2)
     call sla_dcc2s(v2,ra(i),dec(i))
     ra(i) = sla_dranrm(ra(i))
  enddo    
  
end subroutine prenut




subroutine precess_year(ra, dec,mjd, len_bn)
  implicit none
  
  integer, intent(in) :: len_bn
  real*8, intent(in) :: mjd(len_bn)
  real*8, intent(inout) :: ra(len_bn)
  real*8, intent(inout) :: dec(len_bn)

  interface
     real*8 FUNCTION sla_epb(mjddummy)
     real*8 :: mjddummy
     END FUNCTION sla_epb
  end interface

  !f2py integer len_bn
  !f2py real*8 mjd
  !f2py real*8 ra,dec

  integer :: i
  real*8 :: epoch

  do i=1, len_bn
     epoch = sla_epb(mjd(i))
     call sla_preces('FK5', 2000D0, epoch, ra(i), dec(i))
  enddo    
  
end subroutine precess_year





subroutine pa(ra, dec,mjd, lon,lat,pang, len_bn)
  implicit none
  
  integer, intent(in) :: len_bn
  real*8, intent(in) :: lon
  real*8, intent(in) :: lat
  real*8, intent(in) :: mjd(len_bn)
  real*8, intent(in) :: ra(len_bn)
  real*8, intent(in) :: dec(len_bn)
  real*8, intent(out) :: pang(len_bn)

  interface
     real*8 FUNCTION sla_gmst(mjddummy)
     real*8 :: mjddummy
     END FUNCTION sla_gmst
  end interface

  interface
     real*8 FUNCTION sla_pa(radummy, decdummy, phidummy)
     real*8 :: radummy, decdummy, phidummy
     END FUNCTION sla_pa
  end interface

  interface
     real*8 FUNCTION sla_dtt(mjddummy)
     real*8 :: mjddummy
     END FUNCTION sla_dtt
  end interface

  interface
     real*8 FUNCTION sla_eqeqx(mjddummy)
     real*8 :: mjddummy
     END FUNCTION sla_eqeqx
  end interface


  real*8 :: SECPERDAY = 86400.
  real*8 :: mjd_tt
  real*8 :: eq

  !f2py integer len_bn
  !f2py real*8 mjd, lon, lat
  !f2py real*8 ra,dec, pang

  integer :: i
  real*8 :: ha, gmst

  do i=1, len_bn
     gmst = sla_gmst(mjd(i))
     mjd_tt = mjd(i) + sla_dtt(mjd(i))/SECPERDAY
     eq = sla_eqeqx(mjd_tt)
     ha = gmst + lon - ra(i) + eq
     pang(i) = sla_pa(ha, dec(i), lat)
  enddo    

  
end subroutine pa


subroutine e2g(ra, dec, gl, gb, len_bn)
  implicit none
  
  integer, intent(in) :: len_bn
  real*8, intent(in) :: ra(len_bn)
  real*8, intent(in) :: dec(len_bn)
  real*8, intent(out) :: gl(len_bn)
  real*8, intent(out) :: gb(len_bn)

  !f2py integer len_bn
  !f2py real*8 ra,dec,gl,gb

  integer :: i


  do i=1, len_bn
     call sla_eqgal(ra(i), dec(i), gl(i), gb(i))
  enddo    

  
end subroutine e2g

subroutine g2e(gl, gb, ra, dec, len_bn)
  implicit none
  
  integer, intent(in) :: len_bn
  real*8, intent(in) :: gl(len_bn)
  real*8, intent(in) :: gb(len_bn)
  real*8, intent(out) :: ra(len_bn)
  real*8, intent(out) :: dec(len_bn)

  !f2py integer len_bn
  !f2py real*8 ra,dec,gl,gb

  integer :: i


  do i=1, len_bn
     call sla_galeq(gl(i), gb(i), ra(i), dec(i))
  enddo    

  
end subroutine g2e
