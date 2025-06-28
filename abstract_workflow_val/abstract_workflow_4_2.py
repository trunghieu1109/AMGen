async def forward(self, taskInfo):
    from collections import Counter
    # Initialize lists to keep track of sub-tasks and agents
    sub_tasks = []
    agents = []

    '''
    [Stage 1: Core Information and Constraint Extraction]
    
    [Objective]
    - Identify necessary information, inferences, equations, theories, properties, or constraints from the queries and their relationships.
    - Each necessary piece of information corresponds to a distinct sub-task.
    
    [Agent Collaborations]
    - Use Chain-of-Thought (CoT) or Self-Consistency Chain-of-Thought to decompose the problem and extract core components such as equations, constraints, or logical relationships—especially effective for complex problems.
    
    [Examples]
    - Here are many examples of implementing subtasks in this stage (Subtask 1)
    - It could be more than 2 subtasks in this stage, based on your fine-grained task decomposition.
    '''
    
    # --------------------------------------------------------------------------------------------------------------

    # Sub-task 1: Identify the first set of information, constraints, or equations.
    cot_instruction_1 = "Sub-task 1: Identify necessary [information / constraints / equations / theories / relationships #1], with context ...."
    cot_agent_1 = LLMAgentBase(['thinking', 'answer'], 'Chain-of-Thought Agent', model=global_node_model, temperature=0.0)
    thinking1, answer1 = cot_agent_1([taskInfo], cot_instruction_1, is_sub_task=True)
    agents.append(f'CoT agent {cot_agent_1.id}, determine [information / constraints / equations / theories / relationships #1], thinking: {thinking1.content}; answer: {answer1.content}')
    sub_tasks.append(f'Sub-task 1 output: thinking - {thinking1.content}; answer - {answer1.content}')
    
    # --------------------------------------------------------------------------------------------------------------
    
    '''
    [Stage 2: Parametric Representation Derivation]
    
    [Objective]
    - Identify the parametric representation of the quantities in the queries.
    - Each quantity must be handled in a dedicated step.
    
    [Agent Collaborations]
    - Use Chain-of-Thought and Self-Consistency to derive clear and accurate parametric representations for the problem's components.
    - Based on information or constraints from [Stage 1].
    
    [Examples]
    - Here are many examples of implementing subtasks in this stage (Subtask 2 and 3)
    - It could be more than 2 subtasks in this stage, based on your fine-grained task decomposition.
    '''
    
    # --------------------------------------------------------------------------------------------------------------

    # Sub-task 2: Identify parametric representation of [quantity #1] using a single agent.
    cot_instruction_2 = "Sub-task 2: Based on the output from Sub-task 1, identify the parametric representation of [quantity #1], with the following context: ...."
    cot_agent_2 = LLMAgentBase(['thinking', 'answer'], 'Chain-of-Thought Agent', model=global_node_model, temperature=0.0)
    thinking2, answer2 = cot_agent_2([taskInfo, thinking1, answer1], cot_instruction_2, is_sub_task=True)
    agents.append(f'CoT agent {cot_agent_2.id}, identifying parametric representation of [quantity #1], thinking: {thinking2.content}; answer: {answer2.content}')
    sub_tasks.append(f'Sub-task 2 output: thinking - {thinking2.content}; answer - {answer2.content}')

    # Sub-task 3: Identify parametric representation of [quantity #2] using Self-Consistency.
    cot_sc_instruction_3 = "Sub-task 3: Based on the output from Subtask-1 and Sub-task 2, identify the parametric representation of [quantity #2], with the following context: ...."
    N = global_max_sc
    cot_agents_3 = [LLMAgentBase(['thinking', 'answer'], 'Chain-of-Thought Agent', model=global_node_model, temperature=0.5) for _ in range(N)]
    possible_answers_3 = []
    thinking_mapping_3 = {}
    answer_mapping_3 = {}
    
    # Each agent independently generates a potential answer.
    for i in range(N):
        thinking3, answer3 = cot_agents_3[i]([taskInfo, thinking1, answer1, thinking2, answer2], cot_sc_instruction_3, is_sub_task=True)
        agents.append(f'CoT-SC agent {cot_agents_3[i].id}, determining parametric representation of [quantity #2], thinking: {thinking3.content}; answer: {answer3.content}')
        possible_answers_3.append(answer3.content)
        thinking_mapping_3[answer3.content] = thinking3
        answer_mapping_3[answer3.content] = answer3
        
    # The most common answer is chosen for consistency and accuracy.
    answer3_content = Counter(possible_answers_3).most_common(1)[0][0]
    thinking3 = thinking_mapping_3[answer3_content]
    answer3 = answer_mapping_3[answer3_content]
    sub_tasks.append(f'Sub-task 3 output: thinking - {thinking3.content}; answer - {answer3.content}')
    
    # --------------------------------------------------------------------------------------------------------------

    '''
    [Stage 3: Expression Transformation and Alignment]
    
    [Objective]
    - Analyze and transform the identified expressions to obtain a form that better fits the query's conditions.
  
    [Agent Collaborations]
    - Use Reflexion, Debate, or Self-Consistency to refine and align expressions with the specific requirements of the query.
    - Based on parametric representations from [Stage 2].
    
    [Examples]
    - Here are many examples of implementing subtasks in this stage (Subtask 4)
    - It could be more than 2 subtasks in this stage, based on your fine-grained task decomposition.
    '''
    
    # --------------------------------------------------------------------------------------------------------------

    # Sub-task 4: Transform identified expressions into a clearer form that better fits [condition #1].
    cot_reflect_instruction_4 = "Sub-task 4: Based on the outputs from Sub-tasks 1, 2, and 3, transform the [expression #1] into a clearer form that better fits [condition #1]."
    cot_agent_4 = LLMAgentBase(['thinking', 'answer'], 'Chain-of-Thought Agent', model=global_node_model, temperature=0.0)
    critic_agent_4 = LLMAgentBase(['feedback', 'correct'], 'Critic Agent', model=global_node_model, temperature=0.0)
    N_max = global_max_round
    cot_inputs_4 = [taskInfo, thinking1, answer1, thinking2, answer2, thinking3, answer3]
    
    # The CoT agent generates the first version of the transformed expression.
    thinking4, answer4 = cot_agent_4(cot_inputs_4, cot_reflect_instruction_4, 0, is_sub_task=True)
    agents.append(f'Reflexion CoT agent {cot_agent_4.id}, transforming expression [expression #1], thinking: {thinking4.content}; answer: {answer4.content}')

    for i in range(N_max):
        # The critic agent assesses the expression and provides feedback.
        feedback, correct = critic_agent_4([taskInfo, thinking4, answer4], "please review the [expression #1] transformation and correct if needed.", i, is_sub_task=True)
        agents.append(f'Critic agent {critic_agent_4.id}, providing feedback, thinking: {feedback.content}; answer: {correct.content}')
        if correct.content == 'True':
            break
        
        # The CoT agent generates a new version based on the critic's feedback.
        cot_inputs_4.extend([thinking4, answer4, feedback])
        thinking4, answer4 = cot_agent_4(cot_inputs_4, cot_reflect_instruction_4, i + 1, is_sub_task=True)
        agents.append(f'Reflexion CoT agent {cot_agent_4.id}, transform expression [expression #1], thinking: {thinking4.content}; answer: {answer4.content}')
    sub_tasks.append(f'Sub-task 4 output: thinking - {thinking4.content}; answer - {answer4.content}')
    
    # --------------------------------------------------------------------------------------------------------------
    
    '''
    [Stage 4: Final Output Calculation]
    
    [Objective]
    - Calculate the final output corresponding to the query's requirements.
    
    [Agent Collaborations]
    - Use Debate or Reflexion to ensure a powerful and comprehensive final output by interacting with multiple debaters or refining through multiple iterations.
    - Based on expression transformation from [Stage 3]
    
    [Examples]
    - Here are many examples of implementing subtasks in this stage (Subtask 5)
    - It could be more than 2 subtasks in this stage, based on your fine-grained task decomposition.
    '''
    
    # --------------------------------------------------------------------------------------------------------------

    # Sub-task 5: Calculate the [final output]
    debate_instruction_5 = "Sub-task 5: Based on the output of sub-tasks 1, 2, 3, and 4, calculate the [final output], with context ...."
    debate_agents_5 = [LLMAgentBase(['thinking', 'answer'], 'Debate Agent', model=global_node_model, role=role, temperature=0.4) for role in global_debate_role]
    N_max_5 = global_max_round
    all_thinking_5 = [[] for _ in range(N_max_5)]
    all_answer_5 = [[] for _ in range(N_max_5)]
    
    for r in range(N_max_5):
        # Multiple rounds of debate allow agents to build on each other's reasoning.
        for i, agent in enumerate(debate_agents_5):
            input_infos_5 = [taskInfo, thinking4, answer4]
            if r > 0:
                input_infos_5.extend(all_thinking_5[r-1])
            thinking_5, answer_5 = agent(input_infos_5, debate_instruction_5, is_sub_task=True)
            agents.append(f'Debate agent {agent.id}, round {r}, calculating [final output], thinking: {thinking_5.content}; answer: {answer_5.content}')
            all_thinking_5[r].append(thinking_5)
            all_answer_5[r].append(answer_5)
            
    # A final decision agent synthesizes the debate into a single, conclusive answer.
    final_decision_agent_5 = LLMAgentBase(['thinking', 'answer'], 'Final Decision Agent', model=global_node_model, temperature=0.0)
    thinking5, answer5 = final_decision_agent_5([taskInfo] + all_thinking_5[-1] + all_answer_5[-1], "Sub-task 5: Make a final decision on the [final output].", is_sub_task=True)
    agents.append(f'Final Decision agent on calculating [final output], thinking: {thinking5.content}; answer: {answer5.content}')
    sub_tasks.append(f'Sub-task 5 output: thinking - {thinking5.content}; answer - {answer5.content}')
    
    # --------------------------------------------------------------------------------------------------------------

    '''
    [Stage 5: Final Output Verification]
    
    [Objective]
    - Verify the final output to ensure it meets all query conditions and return the verified answer.
    
    [Agent Collaborations]
    - Use Reflexion or Debate patterns to critically assess and confirm the correctness and completeness of the final result.
    - Final output is from [Stage 4]
    
    [Examples]
    - Here are many examples of implementing subtasks in this stage (Subtask 6)
    - It could be more than 2 subtasks in this stage, based on your fine-grained task decomposition.
    '''
    
    # --------------------------------------------------------------------------------------------------------------

    # Sub-task 6: Verify the final answer using an iterative reflection process.
    cot_reflect_instruction_6 = "Sub-task 6: Based on the output from previous subtasks, verify the [final answer] to ensure it matches the [conditions] of the query."
    cot_agent_6 = LLMAgentBase(['thinking', 'answer'], 'Chain-of-Thought Agent', model=global_node_model, temperature=0.0)
    critic_agent_6 = LLMAgentBase(['feedback', 'correct'], 'Critic Agent', model=global_node_model, temperature=0.0)
    N_max_6 = global_max_round
    cot_inputs_6 = [taskInfo, thinking1, answer1, thinking2, answer2, thinking3, answer3, thinking4, answer4, thinking5, answer5]
    
    # The verification agent provides an initial assessment.
    thinking6, answer6 = cot_agent_6(cot_inputs_6, cot_reflect_instruction_6, 0, is_sub_task=True)
    agents.append(f'Reflexion CoT agent {cot_agent_6.id}, verifying [final answer], thinking: {thinking6.content}; answer: {answer6.content}')
    
    for i in range(N_max_6):
        # The critic provides feedback on the verification itself.
        feedback, correct = critic_agent_6([taskInfo, thinking6, answer6], "please review [final output] verification and correct if needed.", i, is_sub_task=True)
        agents.append(f'Critic agent {critic_agent_6.id}, providing feedback on verification, thinking: {feedback.content}; answer: {correct.content}')
        if correct.content == 'True':
            break
        
        # The verification is refined based on the critic's feedback.
        cot_inputs_6.extend([thinking6, answer6, feedback])
        thinking6, answer6 = cot_agent_6(cot_inputs_6, cot_reflect_instruction_6, i + 1, is_sub_task=True)
        agents.append(f'Reflexion CoT agent {cot_agent_6.id}, refining verification of [final answer], thinking: {thinking6.content}; answer: {answer6.content}')
    sub_tasks.append(f'Sub-task 6 output: thinking - {thinking6.content}; answer - {answer6.content}')

    final_answer = await self.make_final_answer(thinking6, answer6, sub_tasks, agents)
    return final_answer