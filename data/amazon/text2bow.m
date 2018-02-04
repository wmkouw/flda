function [S,y] = text2bow(filename, dictionary)
% Extract BoW features based on provided dictionary

% Initialize array
ijv = [];
y = [];

% Start a text scanner
fid = fopen(filename);

lct = 1;
tline = fgetl(fid);
while ischar(tline)
    if rem(lct,10)==0
        disp(['At line ', num2str(lct)]);
    end
    
    % Parse into word:count pairs
    words = strsplit(tline, ' ');
    
    for c = 1:length(words)-1
        
        % Split word:count pair
        word = strsplit(words{c}, ':');
        
        % Find index of word in dictionary
        [~,ix] = find(strcmp(word{1}, dictionary));
        ijv = [ijv; [lct,ix,str2double(word{2})]];
    end
    
    % Add label to label vector
    label = strsplit(words{end}, ':');
    y = [y; label{2}];
    
    % Read line
    tline = fgetl(fid);
    
    % Increment linecounter
    lct = lct + 1;
end

% Generate sparse array from indices
S = sparse(ijv);

% Close file
fclose(fid);

end
