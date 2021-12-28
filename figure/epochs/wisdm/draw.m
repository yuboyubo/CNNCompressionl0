train_path = 'train.txt';
valid_path = 'valid.txt';
test_path = 'test.txt';

train_data = split(fileread(train_path), ",");
valid_data = split(fileread(valid_path), ",");
test_data = split(fileread(test_path), ",");

% initialize
time_seq = size(train_data, 1);

xlim([0 160]) 
ylim([80 100])

train_handle1 = animatedline('LineStyle','-','Color',[0 0.450980392156863 0.741176470588235], 'LineWidth',3);
valid_handle2 = animatedline('LineStyle','-','Color',[1 0 0], 'LineWidth',3);
test_handle3 = animatedline('LineStyle','-','Color','k', 'LineWidth',3);
legend({'Train','Valid','Test'}, 'Location','southeast');

% add points to each figure
for i = 1:150
    addpoints(train_handle1,i,str2double(cell2mat(train_data(i))));
    addpoints(valid_handle2,i,str2double(cell2mat(valid_data(i))));
    addpoints(test_handle3,i,str2double(cell2mat(test_data(i))));
end
ylabel('Accuracy (%)','FontSize',26);

% Create xlabel
xlabel('Epochs','FontSize',26);
set(gcf, 'Color', 'w');
set(gcf, 'Position', [20 20 1200 600]);
set(gca,'Box', 'off','fontsize',26);

% save the plot as pdf format in figure folder
ax = gca;
outerpos = ax.OuterPosition;
ti = ax.TightInset; 
left = outerpos(1) + ti(1);
bottom = outerpos(2) + ti(2);
ax_width = outerpos(3) - ti(1) - ti(3);
ax_height = outerpos(4) - ti(2) - ti(4);
ax.Position = [left bottom ax_width ax_height];
fig = gcf;
fig.PaperPositionMode = 'auto';
fig_pos = fig.PaperPosition;
fig.PaperSize = [fig_pos(3) fig_pos(4)];
print(fig, ['wisdm-epoch-accuracy'],'-dpdf');