function [D,y] = get_amazon(varargin)
% Script to download Amazon sentiment analysis dataset

% Parse
p = inputParser;
addOptional(p, 'save', false);
addOptional(p, 'impute', true);
parse(p, varargin{:});

%% Start downloading files

fprintf('Starting downloads..')

url = 'https://www.cs.jhu.edu/~mdredze/datasets/sentiment/processed_acl.tar.gz';
gunzip(url, '.');
untar('processed_acl.tar')

fprintf('Done \n')

%% Call parse script
[D,y] = parse_amazon_gen('save', p.Results.save);

end
