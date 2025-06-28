async def forward(self, taskInfo):
    from collections import Counter
    # Initialize lists to keep track of sub-tasks and agents
    sub_tasks = []
    agents = []

    '''
    [Stage 1: Parametric Representation Identification]

    [Objective]
    - Identify the parametric representation of the quantities in the queries.
    - Each quantity must be identified through a dedicated reasoning step.

    [Agent Collaborations]
    - Use Chain-of-Thought and Self-Consistency to derive clear and accurate parametric representations for the problem's components.
    
    [Examples]
    - Here are many examples of implementing subtasks in this stage (Subtask 1 and 2)
    - It could be more than 2 subtasks in this stage, based on your fine-grained task decomposition.
    '''

    # --------------------------------------------------------------------------------------------------------------

    # Sub-task 1: Identify parametric representation of [quantity #1] using a single agent.
    cot_instruction = "Sub-task 1: Identify the parametric representation of [quantity #1], with the following context: ...."
    cot_agent = LLMAgentBase(['thinking', 'answer'], 'Chain-of-Thought Agent', model=global_node_model, temperature=0.0)
    thinking1, answer1 = cot_agent([taskInfo], cot_instruction, is_sub_task=True)
    agents.append(f'CoT agent {cot_agent.id}, identifying parametric representation of [quantity #1], thinking: {thinking1.content}; answer: {answer1.content}')
    sub_tasks.append(f'Sub-task 1 output: thinking - {thinking1.content}; answer - {answer1.content}')

    # Sub-task 2: Identify parametric representation of [quantity #2] using Self-Consistency.
    cot_sc_instruction = "Sub-task 2: Based on the output from Sub-task 1, identify the parametric representation of [quantity #2], with the following context: ...."
    N = global_max_sc
    cot_agents = [LLMAgentBase(['thinking', 'answer'], 'Chain-of-Thought Agent', model=global_node_model, temperature=0.5) for _ in range(N)]
    possible_answers = []
    thinking_mapping = {}
    answer_mapping = {}
    
    # Each agent independently generates a potential answer.
    for i in range(N):
        thinking2, answer2 = cot_agents[i]([taskInfo, thinking1, answer1], cot_sc_instruction, is_sub_task=True)
        agents.append(f'CoT-SC agent {cot_agents[i].id}, determining parametric representation of [quantity #2], thinking: {thinking2.content}; answer: {answer2.content}')
        possible_answers.append(answer2.content)
        thinking_mapping[answer2.content] = thinking2
        answer_mapping[answer2.content] = answer2
        
    # The most common answer is chosen for consistency and accuracy.
    answer2_content = Counter(possible_answers).most_common(1)[0][0]
    thinking2 = thinking_mapping[answer2_content]
    answer2 = answer_mapping[answer2_content]
    sub_tasks.append(f'Sub-task 2 output: thinking - {thinking2.content}; answer - {answer2.content}')
    
    # --------------------------------------------------------------------------------------------------------------

    '''
    [Stage 2: Strategy Formulation]

    [Objective]
    - Propose a strategy to thoroughly and comprehensively solve the problem.

    [Agent Collaborations]
    - Use Reflexion, Debate, or Self-Consistency to develop and validate a robust multi-step problem-solving approach.
    - Based on contexts (Parametric representation from [Stage 1])
    
    [Examples]
    - Here are many examples of implementing subtasks in this stage (Subtask 3)
    - It could be more than 2 subtasks in this stage, based on your fine-grained task decomposition.
    '''
    
    # --------------------------------------------------------------------------------------------------------------

    debate_instruction_3 = "Sub-task 3: Based on the output of sub-task 1 and sub-task 2, propose a strategy to thoroughly and comprehensively solve the [problem]"
    debate_agents_3 = [LLMAgentBase(['thinking', 'answer'], 'Debate Agent', model=global_node_model, role=role, temperature=0.5) for role in global_debate_role]
    N_max_3 = global_max_round
    all_thinking_3 = [[] for _ in range(N_max_3)]
    all_answer_3 = [[] for _ in range(N_max_3)]
    for r in range(N_max_3):
        for i, agent in enumerate(debate_agents_3):
            # Generate potential solution of agent i based on previous round's solutions.
            if r == 0:
                thinking3, answer3 = agent([taskInfo, thinking1, answer1, thinking2, answer2], debate_instruction_3, is_sub_task=True)
            else:
                input_infos_3 = [taskInfo, thinking1, answer1, thinking2, answer2] + all_thinking_3[r-1]
                thinking3, answer3 = agent(input_infos_3, debate_instruction_3, is_sub_task=True)
            agents.append(f'Debate agent {agent.id}, round {r}, on the purpose of proposing a strategy to solve the [problem], thinking: {thinking3.content}; answer: {answer3.content}')
            all_thinking_3[r].append(thinking3)
            all_answer_3[r].append(answer3)
    final_decision_agent_3 = LLMAgentBase(['thinking', 'answer'], 'Final Decision Agent', model=global_node_model, temperature=0.0)
    
    # choose the final decision from proposed solutions.
    thinking3, answer3 = final_decision_agent_3([taskInfo] + all_thinking_3[-1] + all_answer_3[-1], "Sub-task 3: Make final decision on [final answer].", is_sub_task=True)
    agents.append(f'Final Decision agent, on the purpose of proposing a strategy to solve the [problem], thinking: {thinking3.content}; answer: {answer3.content}')
    sub_tasks.append(f'Sub-task 3 output: thinking - {thinking3.content}; answer - {answer3.content}')
    
    # --------------------------------------------------------------------------------------------------------------

    '''
    [Stage 3: Final Output Verification]

    [Objective]
    - Verifiy final output to ensure it matches the conditions from queries

    [Agent Collaborations]
    - Use Reflexion, Debate, or Self-Consistency to develop and validate a robust multi-step problem-solving approach.
    - Verify the final output from [Stage 2]
    
    [Examples]
    - Here are many examples of implementing subtasks in this stage (Subtask 4)
    - It could be more than 2 subtasks in this stage, based on your fine-grained task decomposition.
    '''
    
    # --------------------------------------------------------------------------------------------------------------

    # Sub-task 4: Verify the final answer using an iterative reflection process.
    cot_reflect_instruction_4 = "Sub-task 4: Based on the output from Sub-task 3, verify the [final answer] to ensure it matches the [conditions] of the query."
    cot_agent = LLMAgentBase(['thinking', 'answer'], 'Chain-of-Thought Agent', model=global_node_model, temperature=0.0)
    critic_agent = LLMAgentBase(['feedback', 'correct'], 'Critic Agent', model=global_node_model, temperature=0.0)
    N_max_4 = global_max_round
    cot_inputs = [taskInfo, thinking1, answer1, thinking2, answer2, thinking3, answer3]
    
    # The verification agent provides an initial assessment.
    thinking4, answer4 = cot_agent(cot_inputs, cot_reflect_instruction_4, 0, is_sub_task=True)
    agents.append(f'Reflexion CoT agent {cot_agent.id}, verifying [final answer], thinking: {thinking4.content}; answer: {answer4.content}')

    for i in range(N_max_4):
        # The critic provides feedback on the verification itself.
        feedback, correct = critic_agent([taskInfo, thinking4, answer4], "please review [final output] verification and correct if needed.", i, is_sub_task=True)
        agents.append(f'Critic agent {critic_agent.id}, providing feedback on verification, thinking: {feedback.content}; answer: {correct.content}')
        if correct.content == 'True':
            break
        
        # The verification is refined based on the critic's feedback.
        cot_inputs.extend([thinking4, answer4, feedback])
        thinking4, answer4 = cot_agent(cot_inputs, cot_reflect_instruction_4, i + 1, is_sub_task=True)
        agents.append(f'Reflexion CoT agent {cot_agent.id}, refining verification of [final answer], thinking: {thinking4.content}; answer: {answer4.content}')
    sub_tasks.append(f'Sub-task 4 output: thinking - {thinking4.content}; answer - {answer4.content}')

    final_answer = await self.make_final_answer(thinking4, answer4, sub_tasks, agents)
    return final_answer
