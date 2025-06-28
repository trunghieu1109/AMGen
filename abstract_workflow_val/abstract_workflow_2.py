async def forward(self, taskInfo):
    from collections import Counter

    # Initialize lists to keep track of sub-tasks and agents
    sub_tasks = []
    agents = []

    '''
    [Stage 1: Condition Identification and Modular Reasoning]
    
    [Objective]
    - Identify necessary conditions before computing the final output.
    - Each condition should correspond to a distinct subtask for modular reasoning.
    
    [Agent Collaborations]
    - Apply Chain-of-Thought and Self-Consistency Chain-of-Thought approaches to ensure reliable and consistent analysis.
    - Integrate relevant context, task specifications, and outputs from prior subtasks to maintain coherence and consistency.
    
    [Examples]
    - Here are many examples of implementing subtasks in this stage (Subtask 1, Subtask 2)
    - It could be more than 2 subtasks in this stage, based on your fine-grained task decomposition.
    '''

    # --------------------------------------------------------------------------------------------------------------

    # Sub-task 1: Calculate condition #1.
    cot_instruction = "Sub-task 1: Calculate [condition #1], with context ...."
    cot_agent = LLMAgentBase(['thinking', 'answer'], 'Chain-of-Thought Agent', model=self.node_model, temperature=0.0)
    thinking1, answer1 = cot_agent([taskInfo], cot_instruction, is_sub_task=True)
    agents.append(f'CoT agent {cot_agent.id}, calculating [condition #1], thinking: {thinking1.content}; answer: {answer1.content}')
    sub_tasks.append(f'Sub-task 1 output: thinking - {thinking1.content}; answer - {answer1.content}')
    
    print("Subtask 1 answer: ", sub_tasks[-1])

    # Sub-task 2: Calculate condition #2.
    cot_sc_instruction = "Sub-task 2: Based on the output from Sub-task 1, calculate [condition #2], with context ....."
    N = self.max_sc
    cot_agents = [LLMAgentBase(['thinking', 'answer'], 'Chain-of-Thought Agent', model=self.node_model, temperature=0.5) for _ in range(N)]
    possible_answers = []
    thinking_mapping = {}
    answer_mapping = {}

    for i in range(N):
        thinking2, answer2 = cot_agents[i]([taskInfo, thinking1, answer1], cot_sc_instruction, is_sub_task=True)
        agents.append(f'CoT-SC agent {cot_agents[i].id}, determining [condition #2], thinking: {thinking2.content}; answer: {answer2.content}')
        possible_answers.append(answer2.content)
        thinking_mapping[answer2.content] = thinking2
        answer_mapping[answer2.content] = answer2

    answer2 = Counter(possible_answers).most_common(1)[0][0]
    thinking2 = thinking_mapping[answer2]
    answer2 = answer_mapping[answer2]
    sub_tasks.append(f'Sub-task 2 output: thinking - {thinking2.content}; answer - {answer2.content}')
    print("Subtask 2 answer: ", sub_tasks[-1])

    # --------------------------------------------------------------------------------------------------------------

    '''
    [Stage 2: Intermediate Inference and Validation]
    
    [Objective]
    - Compute intermediate output based on conditions.
    - Ensure synthesis is critically evaluated.
    
    [Agent Collaborations]
    - Use Reflexion and Critic agents to iteratively refine and validate outputs.
    - Coordinate agent contributions by cross-referencing both Stage 1 results and real-time deductions within this stage to strengthen final outputs.
    
    [Examples]
    - Here are many examples of implementing subtasks in this stage (Subtask 3)
    - It could be more than 2 subtasks in this stage, based on your fine-grained task decomposition.
    '''

    # --------------------------------------------------------------------------------------------------------------

    # Sub-task 3: Calculate the intermediate output
    cot_reflect_instruction = "Sub-task 3: Based on the outputs from Sub-task 1 and Sub-task 2, calculate the [intermediate output]"
    cot_agent = LLMAgentBase(['thinking', 'answer'], 'Chain-of-Thought Agent', model=self.node_model, temperature=0.0)
    critic_agent = LLMAgentBase(['feedback', 'correct'], 'Critic Agent', model=self.node_model, temperature=0.0)
    N_max = self.max_round

    cot_inputs = [taskInfo, thinking1, answer1, thinking2, answer2]
    thinking3, answer3 = cot_agent(cot_inputs, cot_reflect_instruction, 0, is_sub_task=True)
    agents.append(f'Reflexion CoT agent {cot_agent.id}, calculating [intermediate output], thinking: {thinking3.content}; answer: {answer3.content}')

    for i in range(N_max):
        feedback, correct = critic_agent([taskInfo, thinking3, answer3], "please review the [intermediate output] calculation and correct if needed.", i, is_sub_task=True)
        agents.append(f'Critic agent {critic_agent.id}, providing feedback, thinking: {feedback.content}; answer: {correct.content}')
        if correct.content == 'True':
            break
        cot_inputs.extend([thinking3, answer3, feedback])
        thinking3, answer3 = cot_agent(cot_inputs, cot_reflect_instruction, i + 1, is_sub_task=True)
        agents.append(f'Reflexion CoT agent {cot_agent.id}, refining [intermediate output], thinking: {thinking3.content}; answer: {answer3.content}')
    sub_tasks.append(f'Sub-task 3 output: thinking - {thinking3.content}; answer - {answer3.content}')
    print("Subtask 3 answer: ", sub_tasks[-1])
    

    # --------------------------------------------------------------------------------------------------------------

    '''
    [Stage 3: Final Output Generation and Integration]
    
    [Objective]
    - Transform intermediate output into required format and compute final result.
    
    [Agent Collaborations]
    - Use Debate agents to propose diverse reasoning paths.
    - Use Reflexion to integrate perspectives and reach final conclusion.
    - Transform intermediate output from subtasks in [Stage 2].
    
    [Examples]
    - Here are many examples of implementing subtasks in this stage (Subtask 4)
    - It could be more than 2 subtasks in this stage, based on your fine-grained task decomposition.
    '''

    # --------------------------------------------------------------------------------------------------------------

    # Sub-task 4: Convert intermediate output to specific format and calculate final answer.
    debate_instruction_4 = "Sub-task 4: Based on the output of sub-task 3, convert [intermediate output] into [specific format] and calculate [the final answer]"
    debate_agents_4 = [LLMAgentBase(['thinking', 'answer'], 'Debate Agent', model=self.node_model, role=role, temperature=0.5) for role in self.debate_role]
    N_max_3 = self.max_round

    all_thinking_4 = [[] for _ in range(N_max_3)]
    all_answer_4 = [[] for _ in range(N_max_3)]

    for r in range(N_max_3):
        for agent in debate_agents_4:
            thinking4, answer4 = agent([taskInfo, thinking3, answer3], debate_instruction_4, r, is_sub_task=True)
            agents.append(f'Debate agent {agent.id}, round {r}, converting and calculating final answer, thinking: {thinking4.content}; answer: {answer4.content}')
            all_thinking_4[r].append(thinking4)
            all_answer_4[r].append(answer4)

    # Majority voting on final round
    final_answers = [ans.content for ans in all_answer_4[-1]]
    final_answer_content = Counter(final_answers).most_common(1)[0][0]
    index = final_answers.index(final_answer_content)
    thinking4 = all_thinking_4[-1][index]
    answer4 = all_answer_4[-1][index]
    sub_tasks.append(f'Sub-task 4 output: thinking - {thinking4.content}; answer - {answer4.content}')
    
    print("Subtask 4 answer: ", sub_tasks[-1])

    final_answer = await self.make_final_answer(thinking4, answer4, sub_tasks, agents)
    return final_answer
