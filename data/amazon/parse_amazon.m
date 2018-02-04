function [D,y] = parse_amazon(varargin)
% Parse amazon sentiment .review files into Bag-of-Words encoding

% Parse
p = inputParser;
addOptional(p, 'save', false);
addOptional(p, 'impute', false);
addOptional(p, 'freqThreshold', []);
parse(p, varargin{:});

% Initialize domains
domain_names = {};
domains = [];

try
    load('amazon_dict')
catch
    % Initialize dictionary
    dict = {};
    
    % Find all positive/negative .reviews
    A = dir('processed_acl/*/*e.review');
    
    for a = 1:length(A)
        % Create dictionary from review texts
        dict = generate_dict(join([A(1).folder,'\',A(1).name]), dict);
    end
    save('amazon_dict.mat', 'dict');
end

% Find all positive/negative .reviews
A = dir('processed_acl/*/*e.review');
for a = 1:length(A)
    % Parse text file into sparse array
    [S,l] = text2bow(join([A(1).folder,'\',A(1).name]), dict);
    
    % Store domains
    tmp = strsplit(A(1).folder, '\');
    domain_names{end+1} = tmp{end};
    
    % Store domain sample sizes
    domains = [domains; domains(end)+size(S,1)];
    
    % Append to data array
    D = [D; S];
    y = [y; l];
end

% Remove features with too little occurences
ix = (sum(D,[],1) > p.Results.freqThreshold);
D = D(:,ix);

if p.Results.impute
    D(isnan(D)) = 0;
end

if p.Results.save
    save('amazon', 'D','y', 'domains', 'domain_names');
end

end
