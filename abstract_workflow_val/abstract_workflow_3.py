async def forward(self, taskInfo):
    from collections import Counter
    # Initialize lists to keep track of sub-tasks and agents
    sub_tasks = []
    agents = []

    '''
    [Stage 1: Scenario Exploration and Enumeration]
    
    [Objective]
    - Comprehensively consider or evaluate all possible scenarios of the problems referred to in the queries.
    - Each problem in the queries, if necessary, requires a dedicated step for consideration.
    
    [Agent Collaborations]
    - Use Chain-of-Thought, Self-Consistency Chain-of-Thought, Reflexion, or Debate to enumerate and explore all possible cases of the problems.
    
    [Examples]
    - Here are many examples of implementing subtasks in this stage (Subtask 1 and 2)
    - It could be more than 2 subtasks in this stage, based on your fine-grained task decomposition.
    '''
    
    # --------------------------------------------------------------------------------------------------------------
    
    # Sub-task 1: Consider / calculate all possible cases for [problem #1].
    cot_instruction = "Sub-task 1: Consider / calculate all possible cases of [problem #1], with context ...."
    cot_agent = LLMAgentBase(['thinking', 'answer'], 'Chain-of-Thought Agent', model=self.node_model, temperature=0.0)
    thinking1, answer1 = cot_agent([taskInfo], cot_instruction, is_sub_task=True)
    agents.append(f'CoT agent {cot_agent.id}, consider / calculate all possible scenarios of [problem #1], thinking: {thinking1.content}; answer: {answer1.content}')
    sub_tasks.append(f'Sub-task 1 output: thinking - {thinking1.content}; answer - {answer1.content}')
    
    print("Subtask 1 answer: ", sub_tasks[-1])

    # Sub-task 2: Consider / calculate potential solution of [problem #2]
    cot_sc_instruction = "Sub-task 2: Based on the output from Sub-task 1, consider / calculate potential cases of [problem #2], with context ....."
    N = self.max_sc
    cot_agents = [LLMAgentBase(['thinking', 'answer'], 'Chain-of-Thought Agent', model=self.node_model, temperature=0.5) for _ in range(N)]
    possible_answers = []
    thinking_mapping = {}
    answer_mapping = {}
    for i in range(N):
        # each cot-agent try to calculate all possible cases independently
        thinking2, answer2 = cot_agents[i]([taskInfo, thinking1, answer1], cot_sc_instruction, is_sub_task=True)
        agents.append(f'CoT-SC agent {cot_agents[i].id}, consider all possible cases of [problem #2], thinking: {thinking2.content}; answer: {answer2.content}')
        possible_answers.append(answer2.content)
        thinking_mapping[answer2.content] = thinking2
        answer_mapping[answer2.content] = answer2
        
    # then choose the solutions that appear the most frequently
    answer2 = Counter(possible_answers).most_common(1)[0][0]
    thinking2 = thinking_mapping[answer2]
    answer2 = answer_mapping[answer2]
    sub_tasks.append(f'Sub-task 2 output: thinking - {thinking2.content}; answer - {answer2.content}')
    
    print("Subtask 2 answer: ", sub_tasks[-1])
    
    # --------------------------------------------------------------------------------------------------------------
    
    '''
    [Stage 2: Case Aggregation and Filtering]
    
    [Objective]
    - Aggregate and filter the cases that meet the conditions stated in the queries.
    
    [Agent Collaborations]
    - Use Debate, Reflexion, or Self-Consistency Chain-of-Thought patterns to validate and retain only relevant, condition-satisfying cases.
    - All possible cases derived from [Stage 1].
    
    [Examples]
    - Here are many examples of implementing subtasks in this stage (Subtask 3)
    - It could be more than 2 subtasks in this stage, based on your fine-grained task decomposition.
    '''
    
    # --------------------------------------------------------------------------------------------------------------
    
    # Sub-task 3: Aggregate and filter the valid scenarios that meet the conditions stated in the question.
    cot_reflect_instruction = "Sub-task 3: Based on the outputs from Sub-task 1 and sub-task 2, filter the valid scenarios that meet the [conditions stated in the queries]."
    cot_agent = LLMAgentBase(['thinking', 'answer'], 'Chain-of-Thought Agent', model=self.node_model, temperature=0.0)
    critic_agent = LLMAgentBase(['feedback', 'correct'], 'Critic Agent', model=self.node_model, temperature=0.0)
    N_max = self.max_round
    
    # input for cot-agent
    cot_inputs = [taskInfo, thinking1, answer1, thinking2, answer2]
    
    # generate the first version
    thinking3, answer3 = cot_agent(cot_inputs, cot_reflect_instruction, 0, is_sub_task=True)
    agents.append(f'Reflexion CoT agent {cot_agent.id}, filter valid scenarios of [problem], thinking: {thinking3.content}; answer: {answer3.content}')

    for i in range(N_max):
        # critic-agent debate and criticise pros and cons of previous version
        feedback, correct = critic_agent([taskInfo, thinking3, answer3], "please review the [valid scenarios] filtering and correct if needed.", i, is_sub_task=True)
        agents.append(f'Critic agent {critic_agent.id}, providing feedback, thinking: {feedback.content}; answer: {correct.content}')
        if correct.content == 'True':
            break
        
        # include previous version and feedback from critic-agent as input for cot-agent
        cot_inputs.extend([thinking3, answer3, feedback])
        
        # generate new version based on previous version and feedback
        thinking3, answer3 = cot_agent(cot_inputs, cot_reflect_instruction, i + 1, is_sub_task=True)
        agents.append(f'Reflexion CoT agent {cot_agent.id}, refining valid scenarios of [problem], thinking: {thinking3.content}; answer: {answer3.content}')
    sub_tasks.append(f'Sub-task 3 output: thinking - {thinking3.content}; answer - {answer3.content}')
    
    print("Subtask 3 answer: ", sub_tasks[-1])    
    
    # --------------------------------------------------------------------------------------------------------------
    
    '''
    [Stage 3: Intermediate Output Calculation]
    
    [Objective]
    - Calculate the intermediate output based on the filtered valid cases from [Stage 2].
    
    [Agent Collaborations]
    - Use Debate or Reflexion patterns to derive a more powerful and comprehensive intermediate output.
    - Filterd cases derived from [Stage 2]
    
    [Examples]
    - Here are many examples of implementing subtasks in this stage (Subtask 4)
    - It could be more than 2 subtasks in this stage, based on your fine-grained task decomposition.
    '''
    
    # --------------------------------------------------------------------------------------------------------------

    # Sub-task 4: Calculate the intermediate output
    cot_reflect_instruction = "Sub-task 4: Based on the outputs from Sub-task 1, sub-task 2 and sub-task 3, calculate the [intermediate output]"
    cot_agent = LLMAgentBase(['thinking', 'answer'], 'Chain-of-Thought Agent', model=self.node_model, temperature=0.0)
    critic_agent = LLMAgentBase(['feedback', 'correct'], 'Critic Agent', model=self.node_model, temperature=0.0)
    N_max = self.max_round
    # input for cot-agent
    cot_inputs = [taskInfo, thinking1, answer1, thinking2, answer2]
    
    # generate the first version of [intermediate output]
    thinking4, answer4 = cot_agent(cot_inputs, cot_reflect_instruction, 0, is_sub_task=True)
    agents.append(f'Reflexion CoT agent {cot_agent.id}, calculating [intermediate output], thinking: {thinking4.content}; answer: {answer4.content}')

    for i in range(N_max):
        # critic-agent is reflecting and criticising the previous version
        feedback, correct = critic_agent([taskInfo, thinking4, answer4], "please review the [intermediate output] calculation and correct if needed.", i, is_sub_task=True)
        agents.append(f'Critic agent {critic_agent.id}, providing feedback, thinking: {feedback.content}; answer: {correct.content}')
        if correct.content == 'True':
            break
        cot_inputs.extend([thinking4, answer4, feedback])
        
        # then generate the next version based on previous version and recently feedback
        thinking4, answer4 = cot_agent(cot_inputs, cot_reflect_instruction, i + 1, is_sub_task=True)
        agents.append(f'Reflexion CoT agent {cot_agent.id}, refining [intermediate output], thinking: {thinking4.content}; answer: {answer4.content}')
    sub_tasks.append(f'Sub-task 4 output: thinking - {thinking4.content}; answer - {answer4.content}')
    
    print("Subtask 4 answer: ", sub_tasks[-1])
    
    
    # --------------------------------------------------------------------------------------------------------------
    
    '''
    [Stage 4: Final Answer Derivation]
    
    [Objective]
    - Convert the intermediate output to the required format, then use this to calculate the final answer for the corresponding query.
    
    [Agent Collaborations]
    - Use Reflexion, Self-Consistency, or Debate to ensure a comprehensive and well-supported final answer.
    - Based on intermediate output from [Stage 3].
    
    [Examples]
    - Here are many examples of implementing subtasks in this stage (Subtask 5)
    - It could be more than 2 subtasks in this stage, based on your fine-grained task decomposition.
    '''
    
    # --------------------------------------------------------------------------------------------------------------
    
    # Sub-task 5: Convert the [intermediate output / answer] to [specific format] and calculate [final answer] for corresponding query.
    debate_instruction_5 = "Sub-task 5: Based on the output of sub-task 3, convert [intermediate answer] into [specific format] and calculate [the final answer]"
    debate_agents_5 = [LLMAgentBase(['thinking', 'answer'], 'Debate Agent', model=self.node_model, role=role, temperature=0.5) for role in self.debate_role]
    N_max_5 = self.max_round
    all_thinking_5 = [[] for _ in range(N_max_5)]
    all_answer_5 = [[] for _ in range(N_max_5)]
    for r in range(N_max_5):
        # N_max_5 round of debating
        for i, agent in enumerate(debate_agents_5):
            # each agent propose its solution
            if r == 0:
                thinking_5, answer_5 = agent([taskInfo, thinking_3, answer_3], debate_instruction_5, is_sub_task=True)
            else:
                # generate next solution based on comments and counter-argument from other debaters.
                input_infos_5 = [taskInfo, thinking_3, answer_3] + all_thinking_5[r-1]
                thinking_5, answer_5 = agent(input_infos_5, debate_instruction_5, is_sub_task=True)
            agents.append(f'Debate agent {agent.id}, round {r}, on the purpose of converting [intermediate answer] and calculate [final output], thinking: {thinking_5.content}; answer: {answer_5.content}')
            all_thinking_5[r].append(thinking_5)
            all_answer_5[r].append(answer_5)
    final_decision_agent_5 = LLMAgentBase(['thinking', 'answer'], 'Final Decision Agent', model=self.node_model, temperature=0.0)
    
    # final_decision_agent make final decision of [final output]
    thinking_5, answer_5 = final_decision_agent_5([taskInfo] + all_thinking_5[-1] + all_answer_5[-1], "Sub-task 5: Make final decision on [final output].", is_sub_task=True)
    agents.append(f'Final Decision agent, on the purpose of calculating the [final output], thinking: {thinking_5.content}; answer: {answer_5.content}')
    sub_tasks.append(f'Sub-task 5 output: thinking - {thinking_5.content}; answer - {answer_5.content}')
    
    print("Subtask 5 answer: ", sub_tasks[-1])
    


    final_answer = await self.make_final_answer(thinking5, answer5, sub_tasks, agents)
    return final_answer