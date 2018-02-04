function [dict] = generate_dict(filename, dict)
% Find all unique words in file

% Start a text scanner
fid = fopen(filename);

% Initialize word list
word_list = {};

lct = 1;
tline = fgetl(fid);
while ischar(tline)
    if rem(lct,10)==0
        disp(['At line ', num2str(lct)]);
    end
    
    % Parse into word:count pairs
    wordcount = strsplit(tline, ' ');
    
    for c = 1:length(wordcount)-1
        
         % Parse out word 
         word = strsplit(wordcount{c}, ':');
            
         % Append to word list
         word_list{end+1} = word{1};
    end    
    
    % Read line
    tline = fgetl(fid);
    
    % Increment linecounter
    lct = lct + 1;
end
fclose(fid);

% Remove duplicates
unique_words = unique(word_list);

% Output union of word lists
dict = union(unique_words, dict);

end
