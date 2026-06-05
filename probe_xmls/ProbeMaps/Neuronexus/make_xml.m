%% make xml from scratch from 3 maps

% 1 based
probe = [4,9,3,10,2,11,1,12,5,8,6,7,...
    16,21,15,22,14,23,13,24,17,20,18,19,...
    33,32,34,31,28,37,27,38,26,39,25,40,29,36,30,35,...
    44,49,43,50,42,51,41,52,45,48,46,47,...
    56,61,55,62,54,63,53,64,57,60,58,59]';

probe_0 = probe - ones(size(probe));

% zero based
headstage = [46:-2:16;47:-2:17;[49:2:63,1:2:15];[48:2:62,0:2:14]];
headstage_flip = fliplr(headstage);
%headstage_flipped = flipud(headstage_flip); % this is wrong, when flipped connector, both updown and leftright needs flipping
headstage_flipped = flipud(headstage); % changed 04/20/2024

%H64LP
connector = [37,39,40,42,43,45,46,48,17,19,20,22,23,25,26,28;
             36,38,35,41,34,44,33,47,18,32,21,31,24,30,27,29;
             64,62,60,58,56,54,52,50,15,13,11,9,7,5,3,1;
             63,61,59,57,55,53,51,49,16,14,12,10,8,6,4,2];
connector_0 = connector - ones(size(connector));

% now connector to headstage is (i,j) in headstage to (i,j) in
% connector-flip
xml = [];
for i = 1:size(probe_0,1)
    for j = 1:size(probe_0,2)
        
        [r,c] = find(connector_0 == probe_0(i,j));
        xml(i,j) = headstage_flip(r,c);
    
    end
end

xml_normal = xml';



xml = [];
for i = 1:size(probe_0,1)
    for j = 1:size(probe_0,2)
        
        [r,c] = find(connector_0 == probe_0(i,j));
        xml(i,j) = headstage_flipped(r,c);
    
    end
end

xml_flipped = xml';
