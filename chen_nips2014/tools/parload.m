function varargout = parload(fullpath, varargin)
% par load

varargout = cell(length(varargin), 1);
for ii=1:length(varargin)
    var_name = varargin{ii};
    assert(isa(class(var_name), 'char'), 'only accept name');
    load(fullpath, var_name);
    if exist(var_name, 'var')
        varargout{ii} = eval(var_name);
    else
        varargout{ii} = [];
    end
end
