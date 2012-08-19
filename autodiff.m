classdef autodiff
  properties
    value   %  function value
    deriv   %  derivative value or gradient vector
  end

  methods
    %  class constructor; only the bottom case is needed
    function obj = autodiff(a, b)
      if nargin == 0   %  never intended for use
        obj.value = [];
        obj.deriv = [];
      elseif nargin == 1   %  c = autodiff(a) for variable w/ derivative 1
        obj.value = a;
        obj.deriv = 1;
      else
        obj.value = a;   %  given function value
        obj.deriv = b;   %  given derivative value or gradient vector
      end
    end

    function vec = double(obj)
      vec = [ obj.value, obj.deriv ];
    end

    function h = plus(u, v)
      %  autodiff/mplus overloads + with at least one autodiff
      if     ~isa(u, 'autodiff')   %  u is a scalar
        h = autodiff(u + v.value, v.deriv);
      elseif ~isa(v, 'autodiff')   %  v is a scalar
        h = autodiff(v + u.value, u.deriv);
      else
        h = autodiff(u.value + v.value, u.deriv + v.deriv);
      end
    end

    function h = uminus(u)
      h = autodiff(uminus(u.value), uminus(u.deriv));
    end

    function h = minus(u, v)
      %  autodiff/minus overloads - with at least one autodiff
      if     ~isa(u, 'autodiff')   %  u is a scalar
        h = autodiff(u - v.value, -v.deriv);
      elseif ~isa(v, 'autodiff')   %  v is a scalar
        h = autodiff(u.value - v, -u.deriv);
      else
        h = autodiff(u.value - v.value, u.deriv - v.deriv);
      end
    end

    function h = mtimes(u, v)
      %  autodiff/mtimes overloads * with at least one autodiff
      if     ~isa(u, 'autodiff')   %  u is a scalar
        h = autodiff(u * v.value, u * v.deriv);
      elseif ~isa(v, 'autodiff')   %  v is a scalar
        h = autodiff(v * u.value, v * u.deriv);
      else
        h = autodiff(u.value * v.value, u.deriv * v.value + u.value * v.deriv);
      end
    end

    function h = mrdivide(u, v)
      h = mtimes(u, v ^(-1));
%      %  autodiff/mrdivide overloads / with at least one valder
%      if     ~isa(u, 'autodiff')   %  u is a scalar
%        h = autodiff(u ^ v.value, u ^ v.value * log(u) * v.deriv);
%      elseif ~isa(v, 'autodiff')   %  v is a scalar
%        h = autodiff(u.value ^ v, v * u.value ^ (v - 1) * u.deriv);
%      else
%        h = exp(v * log(u));   %  call overloaded log, * and exp
%      end
    end

    function h = mpower(u, v)
      %  autodiff/mpower overloads ^ with at least one valder
      if     ~isa(u, 'autodiff')   %  u is a scalar
        h = autodiff(u ^ v.value, u ^ v.value * log(u) * v.deriv);
      elseif ~isa(v, 'autodiff')   %  v is a scalar
        h = autodiff(u.value ^ v, v * u.value ^ (v - 1) * u.deriv);
      else
        h = exp(v * log(u));   %  call overloaded log, * and exp
      end
    end

    function h = exp(u)
      h = autodiff(exp(u.value), exp(u.value) * u.deriv);
    end

    function h = log(u)
      h = autodiff(log(u.value), u.deriv / u.value);
    end

    function h = sqrt(u)
      h = autodiff(sqrt(u.value), 1 / (2 * sqrt(u.value)) * u.deriv);
    end

    function h = sin(u)
      h = autodiff(sin(u.value), cos(u.value) * u.deriv);
    end

    function h = cos(u)
      h = autodiff(cos(u.value), -sin(u.value) * u.deriv);
    end

    function h = tan(u)
      h = autodiff(tan(u.value), sec(u.value) ^ 2 * u.deriv);
    end

    function h = asin(u)
      h = autodiff(asin(u.value), 1 / sqrt(1 - u.value ^ 2) * u.deriv);
    end

    function h = acos(u)
      h = autodiff(acos(u.value), -1 / sqrt(1 - u.value ^ 2) * u.deriv);
    end

    function h = atan(u)
      h = autodiff(atan(u.value), 1 / (1 + u.value ^ 2) * u.deriv);
    end

  end
end
