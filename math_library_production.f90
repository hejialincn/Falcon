! math_library_production.f90
module math_library_production
    use, intrinsic :: iso_fortran_env, only: sp=>real32, dp=>real64, qp=>real128, &
                                           int32, int64
    use, intrinsic :: iso_c_binding, only: c_float, c_double
    
    implicit none
    
    private
    public :: Vector2, Vector3, Vector4, Complex, Point2D, Point3D, Quaternion, &
              Matrix4x4, Rectangle, Circle, Sphere, Plane, Random, &
              Statistics, Benchmark, MathConstants, MathUtils, MathFunctions, &
              Vector2f, Vector2d, Vector2i, Vector3f, Vector3d, Vector3i, &
              Vector4f, Vector4d, Vector4i, ComplexF, ComplexD, Point2Df, &
              Point2Dd, Point2Di, Point3Df, Point3Dd, Point3Di, Matrix4x4f, &
              Matrix4x4d, Quaternionf, Quaterniond, Rectanglef, Rectangled, &
              Rectanglei, Circlef, Circled, Spheref, Sphered, Planef, Planed
    
    ! 精度类型定义
    integer, parameter :: SINGLE = sp
    integer, parameter :: DOUBLE = dp
    integer, parameter :: QUAD = qp
    
    ! 数学常量
    type :: MathConstants
    contains
        procedure, nopass :: PI_single => PI_single_const
        procedure, nopass :: PI_double => PI_double_const
        procedure, nopass :: PI_quad => PI_quad_const
        procedure, nopass :: E_single => E_single_const
        procedure, nopass :: E_double => E_double_const
        procedure, nopass :: E_quad => E_quad_const
        procedure, nopass :: EPSILON_single => EPSILON_single_const
        procedure, nopass :: EPSILON_double => EPSILON_double_const
        procedure, nopass :: DEG_TO_RAD_single => DEG_TO_RAD_single_const
        procedure, nopass :: DEG_TO_RAD_double => DEG_TO_RAD_double_const
        procedure, nopass :: RAD_TO_DEG_single => RAD_TO_DEG_single_const
        procedure, nopass :: RAD_TO_DEG_double => RAD_TO_DEG_double_const
    end type MathConstants
    
    type(MathConstants) :: MATH_CONSTANTS
    
    ! 2D向量
    type :: Vector2
        real(sp) :: x, y
    contains
        procedure :: length => vector2_length
        procedure :: length_squared => vector2_length_squared
        procedure :: normalize => vector2_normalize
        procedure :: normalized => vector2_normalized
        procedure :: dot => vector2_dot
        procedure :: cross => vector2_cross
        procedure :: perpendicular => vector2_perpendicular
        procedure :: distance => vector2_distance
        procedure :: distance_squared => vector2_distance_squared
        generic :: operator(+) => add_vector2
        generic :: operator(-) => subtract_vector2
        generic :: operator(*) => multiply_vector2_scalar, multiply_scalar_vector2
        generic :: operator(/) => divide_vector2_scalar
        generic :: assignment(=) => assign_vector2
        procedure :: add_vector2
        procedure :: subtract_vector2
        procedure :: multiply_vector2_scalar
        procedure :: multiply_scalar_vector2
        procedure :: divide_vector2_scalar
        procedure :: assign_vector2
    end type Vector2
    
    ! 3D向量
    type :: Vector3
        real(sp) :: x, y, z
    contains
        procedure :: length => vector3_length
        procedure :: length_squared => vector3_length_squared
        procedure :: normalize => vector3_normalize
        procedure :: normalized => vector3_normalized
        procedure :: dot => vector3_dot
        procedure :: cross => vector3_cross
        procedure :: distance => vector3_distance
        procedure :: distance_squared => vector3_distance_squared
        procedure :: to_vector2 => vector3_to_vector2
        generic :: operator(+) => add_vector3
        generic :: operator(-) => subtract_vector3
        generic :: operator(*) => multiply_vector3_scalar, multiply_scalar_vector3
        generic :: operator(/) => divide_vector3_scalar
        generic :: assignment(=) => assign_vector3
        procedure :: add_vector3
        procedure :: subtract_vector3
        procedure :: multiply_vector3_scalar
        procedure :: multiply_scalar_vector3
        procedure :: divide_vector3_scalar
        procedure :: assign_vector3
    end type Vector3
    
    ! 4D向量
    type :: Vector4
        real(sp) :: x, y, z, w
    contains
        procedure :: length => vector4_length
        procedure :: length_squared => vector4_length_squared
        procedure :: normalize => vector4_normalize
        procedure :: normalized => vector4_normalized
        procedure :: dot => vector4_dot
        procedure :: distance => vector4_distance
        procedure :: distance_squared => vector4_distance_squared
        procedure :: to_vector3 => vector4_to_vector3
        procedure :: to_vector2 => vector4_to_vector2
        generic :: operator(+) => add_vector4
        generic :: operator(-) => subtract_vector4
        generic :: operator(*) => multiply_vector4_scalar, multiply_scalar_vector4
        generic :: operator(/) => divide_vector4_scalar
        generic :: assignment(=) => assign_vector4
        procedure :: add_vector4
        procedure :: subtract_vector4
        procedure :: multiply_vector4_scalar
        procedure :: multiply_scalar_vector4
        procedure :: divide_vector4_scalar
        procedure :: assign_vector4
    end type Vector4
    
    ! 复数
    type :: Complex
        real(sp) :: real, imag
    contains
        procedure :: magnitude => complex_magnitude
        procedure :: magnitude_squared => complex_magnitude_squared
        procedure :: argument => complex_argument
        procedure :: normalize => complex_normalize
        procedure :: normalized => complex_normalized
        procedure :: conjugate => complex_conjugate
        procedure :: exp => complex_exp
        procedure :: log => complex_log
        procedure :: pow => complex_pow
        generic :: operator(+) => add_complex
        generic :: operator(-) => subtract_complex
        generic :: operator(*) => multiply_complex, multiply_complex_scalar, multiply_scalar_complex
        generic :: operator(/) => divide_complex, divide_complex_scalar
        generic :: assignment(=) => assign_complex
        procedure :: add_complex
        procedure :: subtract_complex
        procedure :: multiply_complex
        procedure :: multiply_complex_scalar
        procedure :: multiply_scalar_complex
        procedure :: divide_complex
        procedure :: divide_complex_scalar
        procedure :: assign_complex
    end type Complex
    
    ! 2D点
    type :: Point2D
        real(sp) :: x, y
    contains
        procedure :: distance => point2d_distance
        procedure :: distance_squared => point2d_distance_squared
        procedure :: to_vector => point2d_to_vector
        generic :: operator(-) => subtract_point2d
        generic :: operator(+) => add_point2d_vector, add_vector_point2d
        generic :: operator(-) => subtract_point2d_vector
        generic :: assignment(=) => assign_point2d
        procedure :: subtract_point2d
        procedure :: add_point2d_vector
        procedure :: add_vector_point2d
        procedure :: subtract_point2d_vector
        procedure :: assign_point2d
    end type Point2D
    
    ! 3D点
    type :: Point3D
        real(sp) :: x, y, z
    contains
        procedure :: distance => point3d_distance
        procedure :: distance_squared => point3d_distance_squared
        procedure :: to_vector => point3d_to_vector
        generic :: operator(-) => subtract_point3d
        generic :: operator(+) => add_point3d_vector, add_vector_point3d
        generic :: operator(-) => subtract_point3d_vector
        generic :: assignment(=) => assign_point3d
        procedure :: subtract_point3d
        procedure :: add_point3d_vector
        procedure :: add_vector_point3d
        procedure :: subtract_point3d_vector
        procedure :: assign_point3d
    end type Point3D
    
    ! 四元数
    type :: Quaternion
        real(sp) :: w, x, y, z
    contains
        procedure :: norm => quaternion_norm
        procedure :: norm_squared => quaternion_norm_squared
        procedure :: normalize => quaternion_normalize
        procedure :: normalized => quaternion_normalized
        procedure :: conjugate => quaternion_conjugate
        procedure :: inverse => quaternion_inverse
        procedure :: dot => quaternion_dot
        procedure :: to_rotation_matrix => quaternion_to_rotation_matrix
        procedure :: to_euler_angles => quaternion_to_euler_angles
        generic :: operator(+) => add_quaternion
        generic :: operator(-) => subtract_quaternion
        generic :: operator(*) => multiply_quaternion, multiply_quaternion_scalar, multiply_scalar_quaternion
        generic :: operator(/) => divide_quaternion_scalar
        generic :: assignment(=) => assign_quaternion
        procedure :: add_quaternion
        procedure :: subtract_quaternion
        procedure :: multiply_quaternion
        procedure :: multiply_quaternion_scalar
        procedure :: multiply_scalar_quaternion
        procedure :: divide_quaternion_scalar
        procedure :: assign_quaternion
    end type Quaternion
    
    ! 4x4矩阵
    type :: Matrix4x4
        real(sp) :: data(4,4)
    contains
        procedure :: multiply_matrix => matrix_multiply_matrix
        procedure :: multiply_vector => matrix_multiply_vector
        generic :: operator(*) => multiply_matrix, multiply_vector
        generic :: assignment(=) => assign_matrix
        procedure :: assign_matrix
    end type Matrix4x4
    
    ! 矩形
    type :: Rectangle
        real(sp) :: x, y, width, height
    contains
        procedure :: get_position => rectangle_get_position
        procedure :: get_size => rectangle_get_size
        procedure :: get_area => rectangle_get_area
        procedure :: get_perimeter => rectangle_get_perimeter
        procedure :: get_left => rectangle_get_left
        procedure :: get_right => rectangle_get_right
        procedure :: get_top => rectangle_get_top
        procedure :: get_bottom => rectangle_get_bottom
        procedure :: get_top_left => rectangle_get_top_left
        procedure :: get_top_right => rectangle_get_top_right
        procedure :: get_bottom_left => rectangle_get_bottom_left
        procedure :: get_bottom_right => rectangle_get_bottom_right
        procedure :: get_center => rectangle_get_center
        procedure :: set_position => rectangle_set_position
        procedure :: set_size => rectangle_set_size
        procedure :: contains_point => rectangle_contains_point
        procedure :: contains_rectangle => rectangle_contains_rectangle
        procedure :: intersects => rectangle_intersects
        procedure :: intersection => rectangle_intersection
        procedure :: union_with => rectangle_union_with
        generic :: assignment(=) => assign_rectangle
        procedure :: assign_rectangle
    end type Rectangle
    
    ! 圆
    type :: Circle
        type(Point2D) :: center
        real(sp) :: radius
    contains
        procedure :: get_area => circle_get_area
        procedure :: get_circumference => circle_get_circumference
        procedure :: contains => circle_contains_point
        procedure :: intersects_circle => circle_intersects_circle
        procedure :: intersects_rectangle => circle_intersects_rectangle
        generic :: assignment(=) => assign_circle
        procedure :: assign_circle
    end type Circle
    
    ! 球
    type :: Sphere
        type(Point3D) :: center
        real(sp) :: radius
    contains
        procedure :: get_volume => sphere_get_volume
        procedure :: get_surface_area => sphere_get_surface_area
        procedure :: contains => sphere_contains_point
        procedure :: intersects => sphere_intersects
        generic :: assignment(=) => assign_sphere
        procedure :: assign_sphere
    end type Sphere
    
    ! 平面
    type :: Plane
        type(Vector3) :: normal
        real(sp) :: distance
    contains
        procedure :: distance_to_point => plane_distance_to_point
        procedure :: get_side => plane_get_side
        procedure :: reflect_point => plane_reflect_point
        generic :: assignment(=) => assign_plane
        procedure :: assign_plane
    end type Plane
    
    ! 随机数生成器
    type :: Random
        integer(int64) :: seed
    contains
        procedure :: seed_int => random_seed_int
        procedure :: range_int => random_range_int
        procedure :: range_real => random_range_real
        procedure :: value => random_value
        procedure :: vector2 => random_vector2
        procedure :: vector3 => random_vector3
        procedure :: unit_vector2 => random_unit_vector2
        procedure :: unit_vector3 => random_unit_vector3
        procedure :: boolean => random_boolean
    end type Random
    
    ! 统计函数
    type :: Statistics
    contains
        procedure, nopass :: mean => statistics_mean
        procedure, nopass :: variance => statistics_variance
        procedure, nopass :: standard_deviation => statistics_standard_deviation
        procedure, nopass :: median => statistics_median
        procedure, nopass :: percentile => statistics_percentile
    end type Statistics
    
    ! 基准测试
    type :: Benchmark
        real(dp) :: start_time
        character(len=:), allocatable :: name
        logical :: active
    contains
        procedure :: start => benchmark_start
        procedure :: stop => benchmark_stop
        procedure :: get_duration_microseconds => benchmark_get_duration_microseconds
    end type Benchmark
    
    ! 类型别名
    type(Vector2) :: Vector2f, Vector2d, Vector2i
    type(Vector3) :: Vector3f, Vector3d, Vector3i
    type(Vector4) :: Vector4f, Vector4d, Vector4i
    type(Complex) :: ComplexF, ComplexD
    type(Point2D) :: Point2Df, Point2Dd, Point2Di
    type(Point3D) :: Point3Df, Point3Dd, Point3Di
    type(Matrix4x4) :: Matrix4x4f, Matrix4x4d
    type(Quaternion) :: Quaternionf, Quaterniond
    type(Rectangle) :: Rectanglef, Rectangled, Rectanglei
    type(Circle) :: Circlef, Circled
    type(Sphere) :: Spheref, Sphered
    type(Plane) :: Planef, Planed
    
    ! 数学工具函数接口
    interface is_equal
        module procedure is_equal_single, is_equal_double
    end interface
    
    interface clamp
        module procedure clamp_single, clamp_double, clamp_int
    end interface
    
    interface lerp
        module procedure lerp_single, lerp_double
    end interface
    
    interface smoothstep
        module procedure smoothstep_single, smoothstep_double
    end interface
    
    interface degrees_to_radians
        module procedure degrees_to_radians_single, degrees_to_radians_double
    end interface
    
    interface radians_to_degrees
        module procedure radians_to_degrees_single, radians_to_degrees_double
    end interface
    
    interface sign
        module procedure sign_single, sign_double, sign_int
    end interface
    
    interface fast_inverse_sqrt
        module procedure fast_inverse_sqrt_single, fast_inverse_sqrt_double
    end interface
    
    ! 构造函数接口
    interface Vector2
        module procedure vector2_constructor
    end interface
    
    interface Vector3
        module procedure vector3_constructor
    end interface
    
    interface Vector4
        module procedure vector4_constructor
    end interface
    
    interface Complex
        module procedure complex_constructor
    end interface
    
    interface Point2D
        module procedure point2d_constructor
    end interface
    
    interface Point3D
        module procedure point3d_constructor
    end interface
    
    interface Quaternion
        module procedure quaternion_constructor
    end interface
    
    interface Matrix4x4
        module procedure matrix4x4_constructor
    end interface
    
    interface Rectangle
        module procedure rectangle_constructor
    end interface
    
    interface Circle
        module procedure circle_constructor
    end interface
    
    interface Sphere
        module procedure sphere_constructor
    end interface
    
    interface Plane
        module procedure plane_constructor
    end interface
    
    ! 静态构造函数接口
    interface zero
        module procedure vector2_zero, vector3_zero, vector4_zero, &
                         complex_zero, point2d_zero, point3d_zero, &
                         quaternion_zero, rectangle_zero, circle_zero, &
                         sphere_zero
    end interface
    
    interface one
        module procedure vector2_one, vector3_one, vector4_one, &
                         complex_one, point2d_one, point3d_one
    end interface
    
    interface identity
        module procedure quaternion_identity, matrix4x4_identity
    end interface
    
    ! 数学函数接口
    interface math_sin
        module procedure sin_single, sin_double
    end interface
    
    interface math_cos
        module procedure cos_single, cos_double
    end interface
    
    interface math_tan
        module procedure tan_single, tan_double
    end interface
    
    interface math_asin
        module procedure asin_single, asin_double
    end interface
    
    interface math_acos
        module procedure acos_single, acos_double
    end interface
    
    interface math_atan
        module procedure atan_single, atan_double
    end interface
    
    interface math_atan2
        module procedure atan2_single, atan2_double
    end interface
    
    interface math_exp
        module procedure exp_single, exp_double
    end interface
    
    interface math_log
        module procedure log_single, log_double
    end interface
    
    interface math_log10
        module procedure log10_single, log10_double
    end interface
    
    interface math_pow
        module procedure pow_single, pow_double
    end interface
    
    interface math_sqrt
        module procedure sqrt_single, sqrt_double
    end interface
    
    interface math_abs
        module procedure abs_single, abs_double, abs_int
    end interface
    
    interface math_max
        module procedure max_single, max_double, max_int
    end interface
    
    interface math_min
        module procedure min_single, min_double, min_int
    end interface
    
    interface math_ceil
        module procedure ceil_single, ceil_double
    end interface
    
    interface math_floor
        module procedure floor_single, floor_double
    end interface
    
    interface math_round
        module procedure round_single, round_double
    end interface
    
    interface math_mod
        module procedure mod_single, mod_double
    end interface
    
    interface cubic_interpolate
        module procedure cubic_interpolate_single, cubic_interpolate_double
    end interface
    
contains

    ! 数学常量实现
    real(sp) function PI_single_const()
        PI_single_const = 3.14159265358979323846_sp
    end function
    
    real(dp) function PI_double_const()
        PI_double_const = 3.14159265358979323846_dp
    end function
    
    real(qp) function PI_quad_const()
        PI_quad_const = 3.14159265358979323846_qp
    end function
    
    real(sp) function E_single_const()
        E_single_const = 2.71828182845904523536_sp
    end function
    
    real(dp) function E_double_const()
        E_double_const = 2.71828182845904523536_dp
    end function
    
    real(qp) function E_quad_const()
        E_quad_const = 2.71828182845904523536_qp
    end function
    
    real(sp) function EPSILON_single_const()
        EPSILON_single_const = 1e-6_sp
    end function
    
    real(dp) function EPSILON_double_const()
        EPSILON_double_const = 1e-12_dp
    end function
    
    real(sp) function DEG_TO_RAD_single_const()
        DEG_TO_RAD_single_const = PI_single_const() / 180.0_sp
    end function
    
    real(dp) function DEG_TO_RAD_double_const()
        DEG_TO_RAD_double_const = PI_double_const() / 180.0_dp
    end function
    
    real(sp) function RAD_TO_DEG_single_const()
        RAD_TO_DEG_single_const = 180.0_sp / PI_single_const()
    end function
    
    real(dp) function RAD_TO_DEG_double_const()
        RAD_TO_DEG_double_const = 180.0_dp / PI_double_const()
    end function

    ! 数学工具函数实现
    logical function is_equal_single(a, b, epsilon_val)
        real(sp), intent(in) :: a, b
        real(sp), intent(in), optional :: epsilon_val
        real(sp) :: eps
        
        if (present(epsilon_val)) then
            eps = epsilon_val
        else
            eps = EPSILON_single_const()
        end if
        
        is_equal_single = abs(a - b) <= eps
    end function
    
    logical function is_equal_double(a, b, epsilon_val)
        real(dp), intent(in) :: a, b
        real(dp), intent(in), optional :: epsilon_val
        real(dp) :: eps
        
        if (present(epsilon_val)) then
            eps = epsilon_val
        else
            eps = EPSILON_double_const()
        end if
        
        is_equal_double = abs(a - b) <= eps
    end function
    
    real(sp) function clamp_single(value, min_val, max_val)
        real(sp), intent(in) :: value, min_val, max_val
        clamp_single = min(max_val, max(min_val, value))
    end function
    
    real(dp) function clamp_double(value, min_val, max_val)
        real(dp), intent(in) :: value, min_val, max_val
        clamp_double = min(max_val, max(min_val, value))
    end function
    
    integer function clamp_int(value, min_val, max_val)
        integer, intent(in) :: value, min_val, max_val
        clamp_int = min(max_val, max(min_val, value))
    end function
    
    real(sp) function lerp_single(a, b, t)
        real(sp), intent(in) :: a, b, t
        lerp_single = a + t * (b - a)
    end function
    
    real(dp) function lerp_double(a, b, t)
        real(dp), intent(in) :: a, b, t
        lerp_double = a + t * (b - a)
    end function
    
    real(sp) function smoothstep_single(edge0, edge1, x)
        real(sp), intent(in) :: edge0, edge1, x
        real(sp) :: t
        t = clamp_single((x - edge0) / (edge1 - edge0), 0.0_sp, 1.0_sp)
        smoothstep_single = t * t * (3.0_sp - 2.0_sp * t)
    end function
    
    real(dp) function smoothstep_double(edge0, edge1, x)
        real(dp), intent(in) :: edge0, edge1, x
        real(dp) :: t
        t = clamp_double((x - edge0) / (edge1 - edge0), 0.0_dp, 1.0_dp)
        smoothstep_double = t * t * (3.0_dp - 2.0_dp * t)
    end function
    
    real(sp) function degrees_to_radians_single(degrees)
        real(sp), intent(in) :: degrees
        degrees_to_radians_single = degrees * DEG_TO_RAD_single_const()
    end function
    
    real(dp) function degrees_to_radians_double(degrees)
        real(dp), intent(in) :: degrees
        degrees_to_radians_double = degrees * DEG_TO_RAD_double_const()
    end function
    
    real(sp) function radians_to_degrees_single(radians)
        real(sp), intent(in) :: radians
        radians_to_degrees_single = radians * RAD_TO_DEG_single_const()
    end function
    
    real(dp) function radians_to_degrees_double(radians)
        real(dp), intent(in) :: radians
        radians_to_degrees_double = radians * RAD_TO_DEG_double_const()
    end function
    
    real(sp) function sign_single(x)
        real(sp), intent(in) :: x
        if (x > 0.0_sp) then
            sign_single = 1.0_sp
        else if (x < 0.0_sp) then
            sign_single = -1.0_sp
        else
            sign_single = 0.0_sp
        end if
    end function
    
    real(dp) function sign_double(x)
        real(dp), intent(in) :: x
        if (x > 0.0_dp) then
            sign_double = 1.0_dp
        else if (x < 0.0_dp) then
            sign_double = -1.0_dp
        else
            sign_double = 0.0_dp
        end if
    end function
    
    integer function sign_int(x)
        integer, intent(in) :: x
        if (x > 0) then
            sign_int = 1
        else if (x < 0) then
            sign_int = -1
        else
            sign_int = 0
        end if
    end function
    
    real(sp) function fast_inverse_sqrt_single(x)
        real(sp), intent(in) :: x
        real(sp) :: xhalf
        integer :: i
        real(sp) :: y
        
        if (x == 0.0_sp) then
            fast_inverse_sqrt_single = huge(1.0_sp)
            return
        end if
        
        xhalf = 0.5_sp * x
        i = transfer(x, i)
        i = z'5f3759df' - ishft(i, -1)
        y = transfer(i, y)
        y = y * (1.5_sp - xhalf * y * y) ! 牛顿迭代
        fast_inverse_sqrt_single = y
    end function
    
    real(dp) function fast_inverse_sqrt_double(x)
        real(dp), intent(in) :: x
        ! 对于双精度，使用标准方法
        if (x == 0.0_dp) then
            fast_inverse_sqrt_double = huge(1.0_dp)
        else
            fast_inverse_sqrt_double = 1.0_dp / sqrt(x)
        end if
    end function

    ! Vector2 实现
    type(Vector2) function vector2_constructor(x, y)
        real(sp), intent(in) :: x, y
        vector2_constructor%x = x
        vector2_constructor%y = y
    end function
    
    real(sp) function vector2_length(this)
        class(Vector2), intent(in) :: this
        vector2_length = sqrt(this%x * this%x + this%y * this%y)
    end function
    
    real(sp) function vector2_length_squared(this)
        class(Vector2), intent(in) :: this
        vector2_length_squared = this%x * this%x + this%y * this%y
    end function
    
    subroutine vector2_normalize(this)
        class(Vector2), intent(inout) :: this
        real(sp) :: len, inv_len
        len = this%length()
        if (len <= EPSILON_single_const()) then
            error stop "Cannot normalize zero-length vector"
        end if
        inv_len = fast_inverse_sqrt_single(len * len) * len
        this%x = this%x * inv_len
        this%y = this%y * inv_len
    end subroutine
    
    type(Vector2) function vector2_normalized(this)
        class(Vector2), intent(in) :: this
        vector2_normalized = this
        call vector2_normalized%normalize()
    end function
    
    real(sp) function vector2_dot(this, other)
        class(Vector2), intent(in) :: this, other
        vector2_dot = this%x * other%x + this%y * other%y
    end function
    
    real(sp) function vector2_cross(this, other)
        class(Vector2), intent(in) :: this, other
        vector2_cross = this%x * other%y - this%y * other%x
    end function
    
    type(Vector2) function vector2_perpendicular(this)
        class(Vector2), intent(in) :: this
        vector2_perpendicular%x = -this%y
        vector2_perpendicular%y = this%x
    end function
    
    real(sp) function vector2_distance(this, other)
        class(Vector2), intent(in) :: this, other
        type(Vector2) :: diff
        diff = this - other
        vector2_distance = diff%length()
    end function
    
    real(sp) function vector2_distance_squared(this, other)
        class(Vector2), intent(in) :: this, other
        type(Vector2) :: diff
        diff = this - other
        vector2_distance_squared = diff%length_squared()
    end function
    
    type(Vector2) function add_vector2(this, other)
        class(Vector2), intent(in) :: this, other
        add_vector2%x = this%x + other%x
        add_vector2%y = this%y + other%y
    end function
    
    type(Vector2) function subtract_vector2(this, other)
        class(Vector2), intent(in) :: this, other
        subtract_vector2%x = this%x - other%x
        subtract_vector2%y = this%y - other%y
    end function
    
    type(Vector2) function multiply_vector2_scalar(this, scalar)
        class(Vector2), intent(in) :: this
        real(sp), intent(in) :: scalar
        multiply_vector2_scalar%x = this%x * scalar
        multiply_vector2_scalar%y = this%y * scalar
    end function
    
    type(Vector2) function multiply_scalar_vector2(scalar, this)
        real(sp), intent(in) :: scalar
        class(Vector2), intent(in) :: this
        multiply_scalar_vector2%x = scalar * this%x
        multiply_scalar_vector2%y = scalar * this%y
    end function
    
    type(Vector2) function divide_vector2_scalar(this, scalar)
        class(Vector2), intent(in) :: this
        real(sp), intent(in) :: scalar
        if (abs(scalar) <= EPSILON_single_const()) then
            error stop "Division by zero"
        end if
        divide_vector2_scalar%x = this%x / scalar
        divide_vector2_scalar%y = this%y / scalar
    end function
    
    subroutine assign_vector2(this, other)
        class(Vector2), intent(out) :: this
        type(Vector2), intent(in) :: other
        this%x = other%x
        this%y = other%y
    end subroutine
    
    type(Vector2) function vector2_zero()
        vector2_zero%x = 0.0_sp
        vector2_zero%y = 0.0_sp
    end function
    
    type(Vector2) function vector2_one()
        vector2_one%x = 1.0_sp
        vector2_one%y = 1.0_sp
    end function
    
    type(Vector2) function vector2_unit_x()
        vector2_unit_x%x = 1.0_sp
        vector2_unit_x%y = 0.0_sp
    end function
    
    type(Vector2) function vector2_unit_y()
        vector2_unit_y%x = 0.0_sp
        vector2_unit_y%y = 1.0_sp
    end function

    ! 类似的实现会继续为 Vector3, Vector4, Complex, Point2D, Point3D, Quaternion,
    ! Matrix4x4, Rectangle, Circle, Sphere, Plane, Random, Statistics, Benchmark 等类型
    
    ! 由于代码长度限制，这里只展示了部分实现。完整实现会包含所有类型的完整功能
    
    ! 数学函数实现
    real(sp) function sin_single(angle)
        real(sp), intent(in) :: angle
        sin_single = sin(angle)
    end function
    
    real(dp) function sin_double(angle)
        real(dp), intent(in) :: angle
        sin_double = sin(angle)
    end function
    
    ! 类似的实现会为所有数学函数提供单精度和双精度版本
    
    ! 统计函数实现
    real(sp) function statistics_mean(values)
        real(sp), intent(in) :: values(:)
        if (size(values) == 0) then
            statistics_mean = 0.0_sp
        else
            statistics_mean = sum(values) / size(values)
        end if
    end function
    
    real(sp) function statistics_variance(values)
        real(sp), intent(in) :: values(:)
        real(sp) :: mean_val, sum_sq
        integer :: i
        
        if (size(values) <= 1) then
            statistics_variance = 0.0_sp
            return
        end if
        
        mean_val = statistics_mean(values)
        sum_sq = 0.0_sp
        do i = 1, size(values)
            sum_sq = sum_sq + (values(i) - mean_val) ** 2
        end do
        statistics_variance = sum_sq / (size(values) - 1)
    end function
    
    real(sp) function statistics_standard_deviation(values)
        real(sp), intent(in) :: values(:)
        statistics_standard_deviation = sqrt(statistics_variance(values))
    end function
    
    real(sp) function statistics_median(values)
        real(sp), intent(in) :: values(:)
        real(sp), allocatable :: sorted_values(:)
        integer :: n
        
        if (size(values) == 0) then
            statistics_median = 0.0_sp
            return
        end if
        
        sorted_values = values
        call sort(sorted_values)
        n = size(sorted_values)
        
        if (mod(n, 2) == 0) then
            statistics_median = (sorted_values(n/2) + sorted_values(n/2 + 1)) / 2.0_sp
        else
            statistics_median = sorted_values(n/2 + 1)
        end if
    end function
    
    real(sp) function statistics_percentile(values, p)
        real(sp), intent(in) :: values(:), p
        real(sp), allocatable :: sorted_values(:)
        real(sp) :: index
        integer :: lower, upper
        real(sp) :: weight
        
        if (size(values) == 0) then
            statistics_percentile = 0.0_sp
            return
        end if
        
        sorted_values = values
        call sort(sorted_values)
        index = p * (size(sorted_values) - 1)
        lower = floor(index) + 1
        upper = ceiling(index) + 1
        weight = index - floor(index)
        
        if (lower >= size(sorted_values)) then
            statistics_percentile = sorted_values(size(sorted_values))
        else
            statistics_percentile = sorted_values(lower) * (1.0_sp - weight) + &
                                   sorted_values(upper) * weight
        end if
    end function
    
    ! 辅助排序函数
    subroutine sort(arr)
        real(sp), intent(inout) :: arr(:)
        integer :: i, j
        real(sp) :: temp
        
        do i = 1, size(arr) - 1
            do j = i + 1, size(arr)
                if (arr(i) > arr(j)) then
                    temp = arr(i)
                    arr(i) = arr(j)
                    arr(j) = temp
                end if
            end do
        end do
    end subroutine

end module math_library_production