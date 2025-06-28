async def forward(self, taskInfo):
    from collections import Counter
    
    # Initialize lists to keep track of sub-tasks and agents
    sub_tasks = []
    agents = []

    '''
    [Stage 1: Comprehensive Problem Analysis]
    
    [Objective]
    - Comprehensively consider or evaluate all possible scenarios of the problems referred in queries
    - Each problem in the queries requires a dedicated step for thorough consideration
    
    [Agent Collaborations]
    - Use Chain-of-Thought / Self-Consistency Chain-of-Thought / Reflexion / Debate patterns
    - Integrate relevant context, task specifications, and outputs from prior subtasks to maintain coherence and consistency.
    
    [Examples]
    - Here are many examples of implementing subtasks in this stage (Subtask 1, Subtask 2)
    - It could be more than 2 subtasks in this stage, based on your fine-grained task decomposition.
    '''
    
    # --------------------------------------------------------------------------------------------------------------
    
    # Sub-task 1: Consider/calculate all possible cases for [problem #1]
    cot_instruction = "Sub-task 1: Consider/calculate all possible cases of [problem #1], with context ...."
    cot_agent = LLMAgentBase(['thinking', 'answer'], 'Chain-of-Thought Agent', 
                            model=global_node_model, temperature=0.0)
    thinking1, answer1 = cot_agent([taskInfo], cot_instruction, is_sub_task=True)
    agents.append(f'CoT agent {cot_agent.id}, consider/calculate all possible scenarios of [problem #1], thinking: {thinking1.content}; answer: {answer1.content}')
    sub_tasks.append(f'Sub-task 1 output: thinking - {thinking1.content}; answer - {answer1.content}')

    # Sub-task 2: Consider/calculate potential solutions for [problem #2]
    cot_sc_instruction = "Sub-task 2: Based on the output from Sub-task 1, consider/calculate potential cases of [problem #2], with context ....."
    N = global_max_sc
    cot_agents = [LLMAgentBase(['thinking', 'answer'], 'Chain-of-Thought Agent', 
                              model=global_node_model, temperature=0.5) for _ in range(N)]
    possible_answers = []
    thinking_mapping = {}
    answer_mapping = {}
    
    for i in range(N):
        # Each CoT-SC agent tries to calculate all possible cases independently
        thinking2, answer2 = cot_agents[i]([taskInfo, thinking1, answer1], cot_sc_instruction, is_sub_task=True)
        agents.append(f'CoT-SC agent {cot_agents[i].id}, consider all possible cases of [problem #2], thinking: {thinking2.content}; answer: {answer2.content}')
        possible_answers.append(answer2.content)
        thinking_mapping[answer2.content] = thinking2
        answer_mapping[answer2.content] = answer2
    
    # Choose the solutions that appear most frequently
    answer2 = Counter(possible_answers).most_common(1)[0][0]
    thinking2 = thinking_mapping[answer2]
    answer2 = answer_mapping[answer2]
    sub_tasks.append(f'Sub-task 2 output: thinking - {thinking2.content}; answer - {answer2.content}')
    
    # --------------------------------------------------------------------------------------------------------------
    
    '''
    [Stage 2: Scenario Filtering and Validation]
    
    [Objective]
    - Aggregate and filter cases that meet the conditions stated in the queries
    
    [Agent Collaborations]
    - Use Debate / Reflexion / Self Consistency Chain of Thought patterns
    - Leverage information context from subtasks in [Stage 1] and this stage.
    
    [Examples]
    - Here are many examples of implementing subtasks in this stage (Subtask 3)
    - It could be more than 2 subtasks in this stage, based on your fine-grained task decomposition.
    '''
    
    # --------------------------------------------------------------------------------------------------------------
    
    # Sub-task 3: Aggregate and filter valid scenarios
    cot_reflect_instruction = "Sub-task 3: Based on the outputs from Sub-task 1 and Sub-task 2, filter the valid scenarios that meet the [conditions stated in the queries]."
    cot_agent = LLMAgentBase(['thinking', 'answer'], 'Chain-of-Thought Agent', 
                            model=global_node_model, temperature=0.0)
    critic_agent = LLMAgentBase(['feedback', 'correct'], 'Critic Agent', 
                               model=global_node_model, temperature=0.0)
    N_max = global_max_round
    
    # Input for CoT agent
    cot_inputs = [taskInfo, thinking1, answer1, thinking2, answer2]
    
    # Generate the first version
    thinking3, answer3 = cot_agent(cot_inputs, cot_reflect_instruction, 0, is_sub_task=True)
    agents.append(f'Reflexion CoT agent {cot_agent.id}, filter valid scenarios of [problem], thinking: {thinking3.content}; answer: {answer3.content}')

    for i in range(N_max):
        # Critic agent debates and criticizes pros and cons of previous version
        feedback, correct = critic_agent([taskInfo, thinking3, answer3], 
                                       "please review the [valid scenarios] filtering and correct if needed.", 
                                       i, is_sub_task=True)
        agents.append(f'Critic agent {critic_agent.id}, providing feedback, thinking: {feedback.content}; answer: {correct.content}')
        if correct.content == 'True':
            break
        
        # Include previous version and feedback from critic agent as input
        cot_inputs.extend([thinking3, answer3, feedback])
        
        # Generate new version based on previous version and feedback
        thinking3, answer3 = cot_agent(cot_inputs, cot_reflect_instruction, i + 1, is_sub_task=True)
        agents.append(f'Reflexion CoT agent {cot_agent.id}, refining valid scenarios of [problem], thinking: {thinking3.content}; answer: {answer3.content}')
    sub_tasks.append(f'Sub-task 3 output: thinking - {thinking3.content}; answer - {answer3.content}')
    
    # --------------------------------------------------------------------------------------------------------------
    
    '''
    [Stage 3: Intermediate Output Calculation]
    
    [Objective]  
    - Calculate intermediate output based on filtered scenarios
    
    [Agent Collaborations]
    - Use Debate / Reflexion patterns for comprehensive intermediate output
    - Leverage filtered and valid cases calculated in [Stage 2] and this stage to calculate intermediate output.
    
        [Examples]
    - Here are many examples of implementing subtasks in this stage (Subtask 4)
    - It could be more than 2 subtasks in this stage, based on your fine-grained task decomposition.
    '''
    
    # --------------------------------------------------------------------------------------------------------------

    # Sub-task 4: Calculate intermediate output
    cot_reflect_instruction = "Sub-task 4: Based on the outputs from Sub-task 1, Sub-task 2 and Sub-task 3, calculate the [intermediate output]"
    cot_agent = LLMAgentBase(['thinking', 'answer'], 'Chain-of-Thought Agent', 
                            model=global_node_model, temperature=0.0)
    critic_agent = LLMAgentBase(['feedback', 'correct'], 'Critic Agent', 
                               model=global_node_model, temperature=0.0)
    N_max = global_max_round
    
    # Input for CoT agent
    cot_inputs = [taskInfo, thinking1, answer1, thinking2, answer2, thinking3, answer3]
    
    # Generate first version of intermediate output
    thinking4, answer4 = cot_agent(cot_inputs, cot_reflect_instruction, 0, is_sub_task=True)
    agents.append(f'Reflexion CoT agent {cot_agent.id}, calculating [intermediate output], thinking: {thinking4.content}; answer: {answer4.content}')

    for i in range(N_max):
        # Critic agent reflects and criticizes the previous version
        feedback, correct = critic_agent([taskInfo, thinking4, answer4], 
                                       "please review the [intermediate output] calculation and correct if needed.", 
                                       i, is_sub_task=True)
        agents.append(f'Critic agent {critic_agent.id}, providing feedback, thinking: {feedback.content}; answer: {correct.content}')
        if correct.content == 'True':
            break
        cot_inputs.extend([thinking4, answer4, feedback])
        
        # Generate next version based on previous version and recent feedback
        thinking4, answer4 = cot_agent(cot_inputs, cot_reflect_instruction, i + 1, is_sub_task=True)
        agents.append(f'Reflexion CoT agent {cot_agent.id}, refining [intermediate output], thinking: {thinking4.content}; answer: {answer4.content}')
    sub_tasks.append(f'Sub-task 4 output: thinking - {thinking4.content}; answer - {answer4.content}')
    
    # --------------------------------------------------------------------------------------------------------------
    
    '''
    [Stage 4: Final Answer Generation]
    
    [Objective]
    - Convert intermediate output to specific format and calculate final answer
    
    [Agent Collaborations]
    - Use Reflexion, Self Consistency or Debate for comprehensive answer
    - Leverage calculated intermediate output from subtasks in [Stage 3] to calculate the final answer
    
    [Examples]
    - Here are many examples of implementing subtasks in this stage (Subtask 5)
    - It could be more than 2 subtasks in this stage, based on your fine-grained task decomposition.
    '''
    
    # --------------------------------------------------------------------------------------------------------------
    
    # Sub-task 5: Convert intermediate output and calculate final answer
    debate_instruction_5 = "Sub-task 5: Based on the output of Sub-task 4, convert [intermediate output] into [specific format] and calculate [the final answer]"
    debate_agents_5 = [LLMAgentBase(['thinking', 'answer'], 'Debate Agent', 
                                   model=global_node_model, role=role, temperature=0.5) 
                      for role in global_debate_role]
    N_max_5 = global_max_round
    
    all_thinking_5 = [[] for _ in range(N_max_5)]
    all_answer_5 = [[] for _ in range(N_max_5)]
    
    for r in range(N_max_5):
        # N_max_5 rounds of debating
        for i, agent in enumerate(debate_agents_5):
            # Each agent proposes its solution
            if r == 0:
                thinking_5, answer_5 = agent([taskInfo, thinking4, answer4], 
                                           debate_instruction_5, r, is_sub_task=True)
            else:
                # Generate next solution based on comments and counter-arguments from other debaters
                input_infos_5 = [taskInfo, thinking4, answer4] + all_thinking_5[r-1] + all_answer_5[r-1]
                thinking_5, answer_5 = agent(input_infos_5, debate_instruction_5, r, is_sub_task=True)
            
            agents.append(f'Debate agent {agent.id}, round {r}, converting [intermediate output] and calculating [final output], thinking: {thinking_5.content}; answer: {answer_5.content}')
            all_thinking_5[r].append(thinking_5)
            all_answer_5[r].append(answer_5)
    
    # Final decision agent makes final decision
    final_decision_agent_5 = LLMAgentBase(['thinking', 'answer'], 'Final Decision Agent', 
                                         model=global_node_model, temperature=0.0)
    thinking_5, answer_5 = final_decision_agent_5([taskInfo] + all_thinking_5[-1] + all_answer_5[-1], 
                                                 "Sub-task 5: Make final decision on [final output].", 
                                                 is_sub_task=True)
    agents.append(f'Final Decision agent, calculating [final output], thinking: {thinking_5.content}; answer: {answer_5.content}')
    sub_tasks.append(f'Sub-task 5 output: thinking - {thinking_5.content}; answer - {answer_5.content}')
    
    # --------------------------------------------------------------------------------------------------------------
    
    '''
    [Stage 5: Final Output Verification]
    
    [Objective]
    - Verify correctness of final output against task requirements
    - Double-check with previous intermediate outputs
    
    [Agent Collaborations]
    - Use Reflexion and Debate for verified answer
    - Verify the answer from [Stage 4]
    
    [Examples]
    - Here are many examples of implementing subtasks in this stage (Subtask 6)
    - It could be more than 2 subtasks in this stage, based on your fine-grained task decomposition.
    '''
    
    # --------------------------------------------------------------------------------------------------------------
    
    # Sub-task 6: Verify the final output
    cot_reflect_instruction = "Sub-task 6: Based on the outputs from Sub-task 5, verify the correctness of [final output] against task requirements and previous intermediate outputs"
    cot_agent = LLMAgentBase(['thinking', 'answer'], 'Chain-of-Thought Agent', 
                            model=global_node_model, temperature=0.0)
    critic_agent = LLMAgentBase(['feedback', 'correct'], 'Critic Agent', 
                               model=global_node_model, temperature=0.0)
    N_max = global_max_round
    
    # Input for CoT agent
    cot_inputs = [taskInfo, thinking_5, answer_5, thinking4, answer4]
    
    # Generate first version of verification
    thinking6, answer6 = cot_agent(cot_inputs, cot_reflect_instruction, 0, is_sub_task=True)
    agents.append(f'Reflexion CoT agent {cot_agent.id}, verify [final output], thinking: {thinking6.content}; answer: {answer6.content}')

    for i in range(N_max):
        # Critic agent reflects and criticizes the verification
        feedback, correct = critic_agent([taskInfo, thinking6, answer6], 
                                       "please review [final output] verification and correct if needed.", 
                                       i, is_sub_task=True)
        agents.append(f'Critic agent {critic_agent.id}, providing feedback, thinking: {feedback.content}; answer: {correct.content}')
        if correct.content == 'True':
            break
        cot_inputs.extend([thinking6, answer6, feedback])
        
        # Generate next version based on previous version and recent feedback
        thinking6, answer6 = cot_agent(cot_inputs, cot_reflect_instruction, i + 1, is_sub_task=True)
        agents.append(f'Reflexion CoT agent {cot_agent.id}, refining [final output], thinking: {thinking6.content}; answer: {answer6.content}')
    sub_tasks.append(f'Sub-task 6 output: thinking - {thinking6.content}; answer - {answer6.content}')

    final_answer = await self.make_final_answer(thinking6, answer6, sub_tasks, agents)
    return final_answer


# ,
#     {
#         "name": "abstract_workflow_3_1",
#         "flow": {
#             "Stage 1": {
#                 "Title": "Comprehensive Problem Analysis",
#                 "Content": "This stage comprehensively considers or evaluates all possible scenarios of the problems mentioned in the queries. Each problem receives a dedicated step for a thorough evaluation[1].",
#                 "Objectives": [
#                     "Comprehensively consider or evaluate all possible scenarios of the problems referred in queries[1].",
#                     "Each problem in the queries requires a dedicated step for thorough consideration[1]."
#                 ],
#                 "ExampleSubtasks": [
#                     "Sub-task 1: Consider/calculate all possible cases for [problem #1][1].",
#                     "Sub-task 2: Consider/calculate potential solutions for [problem #2][1]."
#                 ]
#             },
#             "Stage 2": {
#                 "Title": "Scenario Filtering and Validation",
#                 "Content": "This stage aggregates and filters the cases generated in Stage 1 to select only those that meet the specific conditions stated in the queries[1].",
#                 "Objectives": [
#                     "Aggregate and filter cases that meet the conditions stated in the queries[1]."
#                 ],
#                 "ExampleSubtasks": [
#                     "Sub-task 3: Aggregate and filter valid scenarios that meet the [conditions stated in the queries][1]."
#                 ]
#             },
#             "Stage 3": {
#                 "Title": "Intermediate Output Calculation",
#                 "Content": "This stage calculates an intermediate output based on the filtered and validated scenarios identified in Stage 2[1].",
#                 "Objectives": [
#                     "Calculate intermediate output based on filtered scenarios[1]."
#                 ],
#                 "ExampleSubtasks": [
#                     "Sub-task 4: Calculate the [intermediate output] based on the valid scenarios[1]."
#                 ]
#             },
#             "Stage 4": {
#                 "Title": "Final Answer Generation",
#                 "Content": "In this stage, the intermediate output from Stage 3 is converted into a specific format to calculate the final answer[1].",
#                 "Objectives": [
#                     "Convert intermediate output to specific format and calculate final answer[1]."
#                 ],
#                 "ExampleSubtasks": [
#                     "Sub-task 5: Convert [intermediate output] into [specific format] and calculate [the final answer][1]."
#                 ]
#             },
#             "Stage 5": {
#                 "Title": "Final Output Verification",
#                 "Content": "This final stage verifies the correctness of the final output by checking it against the original task requirements and the intermediate results from previous stages[1].",
#                 "Objectives": [
#                     "Verify correctness of final output against task requirements[1].",
#                     "Double-check with previous intermediate outputs[1]."
#                 ],
#                 "ExampleSubtasks": [
#                     "Sub-task 6: Verify the correctness of [final output] against task requirements and previous intermediate outputs[1]."
#                 ]
#             }
#         },
#         "code_path": "abstract_workflow/abstract_workflow_3_1.py"
#     }