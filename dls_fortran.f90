module dls
implicit none
    double precision, parameter :: pi = acos(-1d0)
    double precision, parameter :: sq2pi = sqrt(2d0*pi)
contains
    subroutine trapz(y, x, Nx, int_res)
        integer, intent(in) :: Nx
        double precision, intent(in), dimension(Nx) :: x, y
        double precision, intent(out) :: int_res
        
        integer :: i
        double precision :: int_k, int_kp1

        int_res = 0d0
        int_k = y(1)
        do i=1, Nx-1
            int_kp1 = y(i+1)
            int_res = int_res + (x(i+1) - x(i))*(int_kp1 + int_k)
            int_k = int_kp1
        end do
        int_res = 0.5d0 * int_res
    end subroutine trapz

    subroutine lognormal(x, mu, sig, Nx, px)
        integer, intent(in):: Nx
        double precision, intent(in), dimension(Nx) :: x
        double precision, intent(in) :: mu, sig
        double precision, intent(out), dimension(Nx) :: px

        integer :: i

        do i=1, Nx
            if (x(i) <= 0) then
                px(i) = 0d0
            else
                px(i) = 1d0 / (sq2pi*sig*x(i)) * dexp(-0.5d0*(dlog(x(i)/mu) / sig)**2)
            end if
        end do
    end subroutine lognormal

    subroutine g1(tau, GammaValue, g1_val)
        double precision, intent(in) :: tau, GammaValue
        double precision, intent(out) :: g1_val
        
        g1_val = dexp(-tau*GammaValue)
    end subroutine g1

    subroutine g1_twoG(tau, A1, Gamma1, Gamma2, g1_val)
        double precision, intent(in) :: tau
        double precision, intent(in) :: A1, Gamma1, Gamma2
        double precision, intent(out) :: g1_val
        g1_val = A1*dexp(-tau*Gamma1) + (1d0-A1)*dexp(-tau*Gamma2)
    end subroutine g1_twoG

    subroutine g1_multiG(tau, A, GammaValues, N_Gamma, g1_val)
        double precision, intent(in) :: tau
        double precision, intent(in), dimension(N_Gamma) :: A, GammaValues
        integer, intent(in):: N_Gamma
        double precision, intent(out) :: g1_val
        
        double precision, dimension(N_Gamma) :: p_integrand
        integer :: i

        do i=1, N_Gamma
            p_integrand(i) = A(i)*dexp(-tau*GammaValues(i))
        end do
        call trapz(p_integrand, GammaValues, N_Gamma, g1_val)
    end subroutine g1_multiG

    subroutine g2(B,f,g1, g2_val)
        double precision, intent(in) :: B, f, g1
        double precision, intent(out) :: g2_val
        
        g2_val = B*(1d0 + f*g1**2)
    end subroutine g2
    
    subroutine calc_g2(tau, GammaValue, B, f, N_tau, g2_vals)
        double precision, intent(in), dimension(N_tau) :: tau
        double precision, intent(in) :: GammaValue, B, f
        integer, intent(in) :: N_tau
        double precision, intent(out), dimension(N_tau):: g2_vals
        
        double precision :: g1_val
        integer :: i
        
        do i=1, N_tau
            call g1(tau(i), GammaValue, g1_val)
            call g2(B, f, g1_val, g2_vals(i))
        end do
    end subroutine calc_g2


    subroutine calc_g2_twoG(tau, A1, Gamma1, Gamma2, B, f, N_tau, g2_vals)
        double precision, intent(in), dimension(N_tau) :: tau
        double precision, intent(in) :: A1, Gamma1, Gamma2, B, f
        integer, intent(in) :: N_tau
        double precision, intent(out), dimension(N_tau):: g2_vals
        
        double precision :: g1_val
        integer :: i
        
        do i=1, N_tau
            call g1_twoG(tau(i), A1, Gamma1, Gamma2, g1_val)
            call g2(B, f, g1_val, g2_vals(i))
        end do
    end subroutine calc_g2_twoG

    subroutine calc_g2_multiG(tau, A, GammaValues, B, f, N_tau, N_Gamma, g2_vals)
        integer, intent(in) :: N_tau, N_Gamma
        double precision, intent(in), dimension(N_tau) :: tau
        double precision, intent(in), dimension(N_Gamma) :: A, GammaValues
        double precision, intent(in) :: B, f
        double precision, intent(out), dimension(N_tau):: g2_vals

        double precision :: g1_val
        integer :: i
        
        do i=1, N_tau
            call g1_multiG(tau(i), A, GammaValues, N_Gamma, g1_val)
            call g2(B, f, g1_val, g2_vals(i))
        end do
    end subroutine calc_g2_multiG
end module dls
