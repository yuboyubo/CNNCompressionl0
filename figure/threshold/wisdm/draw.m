% initialize
sparisty = [38.64, 31.68, 28.03, 27.78, 27.29];
accuracy = [95.12, 94.92, 94.65, 94.01, 91.42];

% threold1 = animatedline('','-','Color',[0 0.450980392156863 0.741176470588235], 'LineWidth',3);
% threold2 = animatedline('LineStyle','-','Color',[1 0 0], 'LineWidth',3);
% threold3 = animatedline('LineStyle','-','Color','k', 'LineWidth',3);
% threold4 = animatedline('LineStyle','-','Color','green', 'LineWidth',3);
% threold5 = animatedline('LineStyle','-','Color','blue', 'LineWidth',3);
% legend({'0.005','0.010','0.015','0.020','0.025'}, 'Location','southeast');
fh = figure;
ah = axes(fh);
plot(sparisty,accuracy,'k','LineWidth', 4)
hold(ah,'on');
h(1) = plot(sparisty(1), accuracy(1), '.r','MarkerSize', 60)
hold(ah,'on');
h(2) = plot(sparisty(2), accuracy(2), '.g','MarkerSize', 60)
hold(ah,'on');
h(3) = plot(sparisty(3), accuracy(3), '.b','MarkerSize', 60)
hold(ah,'on');
h(4) = plot(sparisty(4), accuracy(4), '.c','MarkerSize', 60)
hold(ah,'on');
h(5) = plot(sparisty(5), accuracy(5), '.m','MarkerSize', 60)
hold(ah,'on');
ylabel('Accuracy (%)','FontSize',26);

legend(h([1 2 3 4 5]), '0.005','0.010','0.015','0.020','0.025', 'Location','southeast');

xlim([26 40]) 
ylim([90 96])

% Create xlabel
xlabel('Nonzero Parameter (%)','FontSize',26);
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
print(fig,['wisdm-threhold-accuracy'],'-dpdf');