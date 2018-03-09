%%
% initialize grid world
% worldnr = 1;
% worldnr = 2;
% worldnr = 3;
worldnr = 4;
gwinit(worldnr)
%%
% initial world state
state = gwstate();
% initialize a look-up table for Q(s,a) with random values
Q = rand(state.xsize, state.ysize, 4);
% available actions
actions = [1 2 3 4];
% The optimal action is chosen with probability (1-eps), 
% otherwise a random action is chosen
eps = 0.8;
epsorg = eps;
% probability for each action
prob_a = 0.25*[1 1 1 1];
% learning rate
alpha = 0.3;
% discount factor
dc_factor = 0.95;
% k = number of episodes
k=5000;
%%
% for each episode
for i=1:k
    % initialize grid world
    gwinit(worldnr)
    % find initial state
    state = gwstate();
    % loop while not at terminal state
    while(state.isterminal == 0)
        % choose action
        [action, opt_action] = chooseaction(Q, state.pos(1), state.pos(2), actions, prob_a, eps);
        % take action    
        % actions: 
        % 1 - DOWN
        % 2 - UP
        % 3 - RIGHT
        % 4 - LEFT
        next_state = gwaction(action);
        % update Q from feedback
        if(next_state.isvalid == 1)
            r = next_state.feedback;
            Q(state.pos(1), state.pos(2), action) = (1-alpha)*Q(state.pos(1), state.pos(2), action) ...
            + alpha * (r + dc_factor * max(Q(next_state.pos(1),next_state.pos(2),:)));
            state = next_state;
        else
            % Prevent punishing wrong random moves by comparing the
            % direction it moved, and the action taken
            % what direction did it move
            xdiff = next_state.pos(1) - state.pos(1);
            ydiff = next_state.pos(2) - state.pos(2);
            % does the change in position match the taken action
            down = (xdiff == 1 && action == 1);
            up = (xdiff == -1 && action == 2);
            right = (ydiff == 1 && action == 3);
            left = (ydiff == -1 && action == 4);
            % if move was genuine
            if(down || up || right || left)
                Q(state.pos(1), state.pos(2), action) = -inf; 
            else % if random action was taken
                r = -1;
                Q(state.pos(1), state.pos(2), action) = (1-alpha)*Q(state.pos(1), state.pos(2), action) ...
                + alpha * (r + dc_factor * max(Q(next_state.pos(1),next_state.pos(2),:)));
            end
        end
    end
    eps = eps - epsorg/k; % reduce exploration rate
end

figure(1)
gwdraw();
for x = 1:state.xsize
   for y = 1:state.ysize
      [~,I] = max(Q(x,y,:));
      gwplotarrow([x,y],I);
   end
end
% plot Q values
figure(2)
imagesc(Q(:,:,1))

figure(3)
imagesc(Q(:,:,2))

figure(4)
imagesc(Q(:,:,3))

figure(5)
imagesc(Q(:,:,4))
