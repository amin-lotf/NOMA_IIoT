% 5 sbs, 4 sc, 3 sc capacity,  0.6 arrival rate, 0.3 type probabitly
n_runs = 1;
font_size_reward=19;
x_limit = 120*24*60;
x_limit_reward = 20000;
marker_size=8;
face_alpha=0.5;
line_width=2.5;
mappo1_marker = '-o';
mappo2_marker = '--s';
mappo3_marker = ':^';
mappo4_marker = '-v';

edge_alpha= 0.18;
mappo1_color='#34abeb';
mappo2_color='#f53242';
mappo3_color='#a305ff';
mappo4_color='#2cdb04';
mappo5_color='#f2906f';
mappo6_color='#f39f6f';
load_info

legend_names=["\textbf{Dual MAPPO (RNN + ResNet)}","\textbf{Dual MAPPO (MLP)}","\textbf{Single MAPPO  (RNN + ResNet)}"];

x_labels={'\boldmath $1$';'\boldmath $10$'};

bar_legend_names=categorical(x_labels);
l01=[3.32;3.02;2.73];
l10=[3.25;2.94;2.53];



% ep_i=(ep(1)-ep(3))/ep(1)
% bits_i=(bits(3)-bits(1))/bits(1)
h15=figure(15);
clf(h15,'reset');
x = bar_legend_names;
x = reordercats(x,cellstr(x)');

nil = [0; 0 ; 0 ];
b=bar(x, [l01  l10]);
for i = 1:numel(b)
    % Get the X- and Y-coordinates at the top of each bar
    xtips = b(i).XEndPoints;
    ytips = b(i).YEndPoints;
    
    % Convert the YData into a cell array of strings (or use string(...) directly)
    labels = string(b(i).YData);
    
    % Place text centered horizontally and just above each bar
    text(xtips, ytips, labels, ...
         'HorizontalAlignment','center', ...
         'VerticalAlignment','bottom',...
         'FontSize',11);
end
ylabel("\textbf{EP (Mbits/J)}",'Interpreter','latex','FontSize', font_size_print);

 % ylim([0 4.0]);
% yyaxis right
b(1).FaceColor = '#0072B2';
b(2).FaceColor = '#D55E00';
b(3).FaceColor = '#009E73';
% b=bar(x, [nil l10],'FaceColor','flat');
% ylabel("\textbf{Processed tasks (Mbits)}",'Interpreter','latex','FontSize', font_size_print);
xlabel("\boldmath $\hat{L}$",'Interpreter','latex','FontSize', font_size_reward);
grid on
b(1).FaceColor = '#0072B2';
b(2).FaceColor = '#D55E00';
b(3).FaceColor = '#009E73';
set(gca,'ycolor','k') 
set(gca,'FontSize',font_size_print_m)
set(gca, 'XTickLabel', x_labels, 'TickLabelInterpreter', 'latex');
ylim([0 4.5]);
legend(legend_names,'Interpreter','latex','Location','northeast','FontSize', font_size_print_m,'NumColumns',1);
% saveas(h15,'graphs/ec_rate_2_bar.png');