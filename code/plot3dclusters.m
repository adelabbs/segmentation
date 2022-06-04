% PLOTCLUSTERS Plots a set of differently colored point clusters.
%   PLOT3DCLUSTERS( DATA, LABELS, MEANS ) plots the 3D data D with each of
%   its different clusters as different colors. The cluster means and
%   labels are specified by LABELS and MEANS respectively.


function plotclusters( data, labels, means )

% plot each cluster
n = size(means,2);
for label = 1:n
    % pick random color
    color = rand([3 1]);
    cluster = data( :, find( labels == label ) );
    plot3(cluster(1,:),cluster(2,:),cluster(3,:),'.','Color',color); hold on;
    plot3(means(1,label), means(2,label), means(3,label), 'kx', 'MarkerSize', 24.0, 'LineWidth', 4.0);
end
grid on;
    
    