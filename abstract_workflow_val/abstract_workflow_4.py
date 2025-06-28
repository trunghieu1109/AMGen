async def forward(self, taskInfo):
    from collections import Counter
    # Initialize lists to keep track of sub-tasks and agents
    sub_tasks = []
    agents = []

    '''
    [Stage 1: Parametric Representation Identification]

    [Objective]
    - Identify the parametric representation of the quantities in the queries.
    - Ensure each quantity is addressed through a dedicated step for clarity and modularity.

    [Agent Collaborations]
    - Use Chain-of-Thought and Self-Consistency to derive clear, structured parametric representations of the problem's components.
    
    [Examples]
    - Here are many examples of implementing subtasks in this stage (Subtask 1 and 2)
    - It could be more than 2 subtasks in this stage, based on your fine-grained task decomposition.
    '''
    
    # --------------------------------------------------------------------------------------------------------------

    # Sub-task 1: Identify parametric representation of [quantity #1] using a single agent.
    cot_instruction = "Sub-task 1: Identify the parametric representation of [quantity #1], with the following context: ...."
    cot_agent = LLMAgentBase(['thinking', 'answer'], 'Chain-of-Thought Agent', model=self.node_model, temperature=0.0)
    thinking1, answer1 = cot_agent([taskInfo], cot_instruction, is_sub_task=True)
    agents.append(f'CoT agent {cot_agent.id}, identifying parametric representation of [quantity #1], thinking: {thinking1.content}; answer: {answer1.content}')
    sub_tasks.append(f'Sub-task 1 output: thinking - {thinking1.content}; answer - {answer1.content}')
    print("Subtask 1 answer: ", sub_tasks[-1])
    

    # Sub-task 2: Identify parametric representation of [quantity #2] using Self-Consistency.
    cot_sc_instruction = "Sub-task 2: Based on the output from Sub-task 1, identify the parametric representation of [quantity #2], with the following context: ...."
    N = self.max_sc
    cot_agents = [LLMAgentBase(['thinking', 'answer'], 'Chain-of-Thought Agent', model=self.node_model, temperature=0.5) for _ in range(N)]
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
    print("Subtask 2 answer: ", sub_tasks[-1])
    
    
    # --------------------------------------------------------------------------------------------------------------

    '''
    [Stage 2: Expression Analysis and Transformation]

    [Objective]
    - Analyze and transform the identified expressions to obtain a form that better fits the query's conditions.
    
    [Agent Collaborations]
    - Use Reflexion, Debate, or Self-Consistency to refine and adapt the expressions for alignment with the query's specific requirements.
    - Based on identified expressions from [Stage 1].
    
    [Examples]
    - Here are many examples of implementing subtasks in this stage (Subtask 3)
    - It could be more than 2 subtasks in this stage, based on your fine-grained task decomposition.
    '''
    
    # --------------------------------------------------------------------------------------------------------------

    # Sub-task 3: Transform identified expressions into a clearer form that better fits [condition #1].
    cot_reflect_instruction = "Sub-task 3: Based on the outputs from Sub-task 1 and Sub-task 2, transform the [expression #1] into a clearer form that better fits [condition #1]."
    cot_agent = LLMAgentBase(['thinking', 'answer'], 'Chain-of-Thought Agent', model=self.node_model, temperature=0.0)
    critic_agent = LLMAgentBase(['feedback', 'correct'], 'Critic Agent', model=self.node_model, temperature=0.0)
    N_max = self.max_round
    cot_inputs = [taskInfo, thinking1, answer1, thinking2, answer2]
    
    # The CoT agent generates the first version of the transformed expression.
    thinking3, answer3 = cot_agent(cot_inputs, cot_reflect_instruction, 0, is_sub_task=True)
    agents.append(f'Reflexion CoT agent {cot_agent.id}, transforming expression [expression #1], thinking: {thinking3.content}; answer: {answer3.content}')

    for i in range(N_max):
        # The critic agent assesses the expression and provides feedback.
        feedback, correct = critic_agent([taskInfo, thinking3, answer3], "please review the [expression #1] transformation and correct if needed.", i, is_sub_task=True)
        agents.append(f'Critic agent {critic_agent.id}, providing feedback, thinking: {feedback.content}; answer: {correct.content}')
        if correct.content == 'True':
            break
        
        # The CoT agent generates a new version based on the critic's feedback.
        cot_inputs.extend([thinking3, answer3, feedback])
        thinking3, answer3 = cot_agent(cot_inputs, cot_reflect_instruction, i + 1, is_sub_task=True)
        agents.append(f'Reflexion CoT agent {cot_agent.id}, transform expression [expression #1], thinking: {thinking3.content}; answer: {answer3.content}')
    sub_tasks.append(f'Sub-task 3 output: thinking - {thinking3.content}; answer - {answer3.content}')
    print("Subtask 3 answer: ", sub_tasks[-1])
    

    # --------------------------------------------------------------------------------------------------------------

    '''
    [Stage 3: Final Output Calculation]

    [Objective]
    - Calculate the final output corresponding to the query's requirements.

    [Agent Collaborations]
    - Use Debate or Reflexion to ensure a powerful and comprehensive final output by interacting with multiple debaters or refining through multiple iterations.
    - Based on expression transformations from [Stage 2]
    
    [Examples]
    - Here are many examples of implementing subtasks in this stage (Subtask 4)
    - It could be more than 2 subtasks in this stage, based on your fine-grained task decomposition.
    '''
    
    # --------------------------------------------------------------------------------------------------------------

    # Sub-task 4: Calculate the [final output]
    debate_instruction_4 = "Sub-task 4: Based on the output of sub-tasks 1, 2, and 3, calculate the [final output], with context ...."
    debate_agents_4 = [LLMAgentBase(['thinking', 'answer'], 'Debate Agent', model=self.node_model, role=role, temperature=0.4) for role in self.debate_role]
    N_max_4 = self.max_round
    all_thinking_4 = [[] for _ in range(N_max_4)]
    all_answer_4 = [[] for _ in range(N_max_4)]
    
    for r in range(N_max_4):
        # Multiple rounds of debate allow agents to build on each other's reasoning.
        for i, agent in enumerate(debate_agents_4):
            input_infos_4 = [taskInfo, thinking3, answer3]
            if r > 0:
                input_infos_4.extend(all_thinking_4[r-1])
            thinking_4, answer_4 = agent(input_infos_4, debate_instruction_4, is_sub_task=True)
            agents.append(f'Debate agent {agent.id}, round {r}, calculating [final output], thinking: {thinking_4.content}; answer: {answer_4.content}')
            all_thinking_4[r].append(thinking_4)
            all_answer_4[r].append(answer_4)
            
    # A final decision agent synthesizes the debate into a single, conclusive answer.
    final_decision_agent_4 = LLMAgentBase(['thinking', 'answer'], 'Final Decision Agent', model=self.node_model, temperature=0.0)
    thinking_4, answer_4 = final_decision_agent_4([taskInfo] + all_thinking_4[-1] + all_answer_4[-1], "Sub-task 4: Make a final decision on the [final output].", is_sub_task=True)
    agents.append(f'Final Decision agent on calculating [final output], thinking: {thinking_4.content}; answer: {answer_4.content}')
    sub_tasks.append(f'Sub-task 4 output: thinking - {thinking_4.content}; answer - {answer_4.content}')
    print("Subtask 4 answer: ", sub_tasks[-1])
    
    
    # --------------------------------------------------------------------------------------------------------------

    '''
    [Stage 4: Final Output Verification]

    [Objective]
    - Verify the final output to ensure it meets all query conditions and return the verified answer.

    [Agent Collaborations]
    - Use Reflexion or Debate patterns to critically validate the final result against both the task requirements and intermediate reasoning steps.
    - Verify final output from [Stage 3]
    
    [Examples]
    - Here are many examples of implementing subtasks in this stage (Subtask 5)
    - It could be more than 2 subtasks in this stage, based on your fine-grained task decomposition.
    '''
    
    # --------------------------------------------------------------------------------------------------------------

    # Sub-task 5: Verify the final answer using an iterative reflection process.
    cot_reflect_instruction_5 = "Sub-task 5: Based on the output from Sub-task 4, verify the [final answer] to ensure it matches the [conditions] of the query."
    cot_agent = LLMAgentBase(['thinking', 'answer'], 'Chain-of-Thought Agent', model=self.node_model, temperature=0.0)
    critic_agent = LLMAgentBase(['feedback', 'correct'], 'Critic Agent', model=self.node_model, temperature=0.0)
    N_max_5 = self.max_round
    cot_inputs = [taskInfo, thinking1, answer1, thinking2, answer2, thinking_4, answer_4]
    
    # The verification agent provides an initial assessment.
    thinking5, answer5 = cot_agent(cot_inputs, cot_reflect_instruction_5, 0, is_sub_task=True)
    agents.append(f'Reflexion CoT agent {cot_agent.id}, verifying [final answer], thinking: {thinking5.content}; answer: {answer5.content}')

    for i in range(N_max_5):
        # The critic provides feedback on the verification itself.
        feedback, correct = critic_agent([taskInfo, thinking5, answer5], "please review [final output] verification and correct if needed.", i, is_sub_task=True)
        agents.append(f'Critic agent {critic_agent.id}, providing feedback on verification, thinking: {feedback.content}; answer: {correct.content}')
        if correct.content == 'True':
            break
        
        # The verification is refined based on the critic's feedback.
        cot_inputs.extend([thinking5, answer5, feedback])
        thinking5, answer5 = cot_agent(cot_inputs, cot_reflect_instruction_5, i + 1, is_sub_task=True)
        agents.append(f'Reflexion CoT agent {cot_agent.id}, refining verification of [final answer], thinking: {thinking5.content}; answer: {answer5.content}')
    sub_tasks.append(f'Sub-task 5 output: thinking - {thinking5.content}; answer - {answer5.content}')
    
    print("Subtask 5 answer: ", sub_tasks[-1])
    

    final_answer = await self.make_final_answer(thinking5, answer5, sub_tasks, agents)
    return final_answer
