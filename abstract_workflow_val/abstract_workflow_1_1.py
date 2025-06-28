async def forward(self, taskInfo):
   
    from collections import Counter
    
    # Initialize lists to keep track of sub-tasks and agents
    sub_tasks = []
    agents = []

    '''
    [Stage 1: Expression Analysis and Data Interpretation]
    
    [Objective]
    - Analyze all expressions and data present in the query systematically
    - For each expression, determine behavior, range, visualization characteristics, and relationships
    - Ensure modular reasoning through distinct subtasks for each analytical component
    
    [Agent Collaborations]
    - Use Chain-of-Thought / Self-Consistency Chain-of-Thought patterns for step-by-step reliability
    - Maintain contextual coherence through chaining from previous subtask outputs
    
    [Examples]
    - Here are many examples of implementing subtasks in this stage (Subtask 1, Subtask 2)
    - It could be more than 2 subtasks in this stage, based on your fine-grained task decomposition.
    '''
    
    # --------------------------------------------------------------------------------------------------------------
    
    # Sub-task 1: Analyze primary expression/data component
    cot_instruction = "Sub-task 1: Analyze [expression #1], determining its behavior, range, and key characteristics with context from [taskInfo]"
    cot_agent = LLMAgentBase(['thinking', 'answer'], 'Chain-of-Thought Agent', 
                            model=self.node_model, temperature=0.0)
    thinking1, answer1 = cot_agent([taskInfo], cot_instruction, is_sub_task=True)
    agents.append(f'CoT agent {cot_agent.id}, analyzing [expression #1], thinking: {thinking1.content}; answer: {answer1.content}')
    sub_tasks.append(f'Sub-task 1 output: thinking - {thinking1.content}; answer - {answer1.content}')
    
    print("Subtask 1: ", sub_tasks[-1])

    # Sub-task 2: Analyze secondary expression/data component with self-consistency
    cot_sc_instruction = "Sub-task 2: Based on Sub-task 1 output, analyze [expression #2] and its relationship to the first expression"
    N = self.max_sc
    cot_agents = [LLMAgentBase(['thinking', 'answer'], 'Chain-of-Thought Agent', 
                              model=self.node_model, temperature=0.5) for _ in range(N)]
    possible_answers = []
    thinking_mapping = {}
    answer_mapping = {}
    
    for i in range(N):
        # Each CoT-SC agent analyzes independently for consensus building
        thinking2, answer2 = cot_agents[i]([taskInfo, thinking1, answer1], cot_sc_instruction, is_sub_task=True)
        agents.append(f'CoT-SC agent {cot_agents[i].id}, analyzing [expression #2], thinking: {thinking2.content}; answer: {answer2.content}')
        possible_answers.append(answer2.content)
        thinking_mapping[answer2.content] = thinking2
        answer_mapping[answer2.content] = answer2
    
    # Select most consistent analysis through voting
    answer2 = Counter(possible_answers).most_common(1)[0][0]
    thinking2 = thinking_mapping[answer2]
    answer2 = answer_mapping[answer2]
    sub_tasks.append(f'Sub-task 2 output: thinking - {thinking2.content}; answer - {answer2.content}')
    print("Subtask 2: ", sub_tasks[-1])
    
    
    # --------------------------------------------------------------------------------------------------------------
    
    '''
    [Stage 2: Intermediate Computation and Synthesis]
    
    [Objective]
    - Synthesize insights from expression analysis to compute intermediate values
    - Apply critical reasoning to validate computational steps
    - Build foundation for final answer generation
    
    [Agent Collaborations]
    - Use Reflexion pattern with Critic feedback for iterative refinement
    - Integrate all context from Stage 1 analysis for comprehensive synthesis
    
    [Examples]
    - Here are many examples of implementing subtasks in this stage (Subtask 3)
    - It could be more than 2 subtasks in this stage, based on your fine-grained task decomposition.
    '''
    
    # --------------------------------------------------------------------------------------------------------------
    
    # Sub-task 3: Calculate intermediate output with reflexion
    cot_reflect_instruction = "Sub-task 3: Based on Sub-task 1 and Sub-task 2 outputs, calculate intermediate values and synthesize key insights"
    cot_agent = LLMAgentBase(['thinking', 'answer'], 'Chain-of-Thought Agent', 
                            model=self.node_model, temperature=0.0)
    critic_agent = LLMAgentBase(['feedback', 'correct'], 'Critic Agent', 
                               model=self.node_model, temperature=0.0)
    N_max = self.max_round
    
    # Input aggregation from previous stages
    cot_inputs = [taskInfo, thinking1, answer1, thinking2, answer2]
    
    # Generate initial intermediate computation
    thinking3, answer3 = cot_agent(cot_inputs, cot_reflect_instruction, 0, is_sub_task=True)
    agents.append(f'Reflexion CoT agent {cot_agent.id}, calculating intermediate output, thinking: {thinking3.content}; answer: {answer3.content}')

    # Iterative refinement through critic feedback
    for i in range(N_max):
        feedback, correct = critic_agent([taskInfo, thinking3, answer3], 
                                       "Review intermediate output calculation for accuracy and completeness", 
                                       i, is_sub_task=True)
        agents.append(f'Critic agent {critic_agent.id}, providing feedback, thinking: {feedback.content}; answer: {correct.content}')
        if correct.content == 'True':
            break
        
        # Incorporate feedback for next iteration
        cot_inputs.extend([thinking3, answer3, feedback])
        thinking3, answer3 = cot_agent(cot_inputs, cot_reflect_instruction, i + 1, is_sub_task=True)
        agents.append(f'Reflexion CoT agent {cot_agent.id}, refining intermediate output, thinking: {thinking3.content}; answer: {answer3.content}')
    
    sub_tasks.append(f'Sub-task 3 output: thinking - {thinking3.content}; answer - {answer3.content}')
    
    print("Subtask 3: ", sub_tasks[-1])
    
    
    # --------------------------------------------------------------------------------------------------------------
    
    '''
    [Stage 3: Final Answer Generation and Integration]
    
    [Objective]
    - Integrate all analytical and computational results into final conclusion
    - Ensure logical consistency and completeness across all processing stages
    - Generate final answer that addresses the original query comprehensively
    
    [Agent Collaborations]
    - Apply Reflexion and Debate patterns for final answer validation
    - Leverage complete context chain from all previous stages
    
    [Examples]
    - Here are many examples of implementing subtasks in this stage (Subtask 4)
    - It could be more than 2 subtasks in this stage, based on your fine-grained task decomposition.
    '''
    
    # --------------------------------------------------------------------------------------------------------------
    
    # Sub-task 4: Generate final answer with comprehensive integration
    cot_reflect_instruction = "Sub-task 4: Integrate outputs from all previous subtasks to generate the final answer for the query"
    cot_agent = LLMAgentBase(['thinking', 'answer'], 'Chain-of-Thought Agent', 
                            model=self.node_model, temperature=0.0)
    critic_agent = LLMAgentBase(['feedback', 'correct'], 'Critic Agent', 
                               model=self.node_model, temperature=0.0)
    N_max = self.max_round
    
    # Complete context integration from all stages
    cot_inputs = [taskInfo, thinking1, answer1, thinking2, answer2, thinking3, answer3]
    
    # Generate initial final answer
    thinking4, answer4 = cot_agent(cot_inputs, cot_reflect_instruction, 0, is_sub_task=True)
    agents.append(f'Reflexion CoT agent {cot_agent.id}, generating final answer, thinking: {thinking4.content}; answer: {answer4.content}')

    # Final validation and refinement loop
    for i in range(N_max):
        feedback, correct = critic_agent([taskInfo, thinking4, answer4], 
                                       "Review final answer for completeness, accuracy, and alignment with original query", 
                                       i, is_sub_task=True)
        agents.append(f'Critic agent {critic_agent.id}, providing feedback, thinking: {feedback.content}; answer: {correct.content}')
        if correct.content == 'True':
            break
        
        # Final refinement based on critic feedback
        cot_inputs.extend([thinking4, answer4, feedback])
        thinking4, answer4 = cot_agent(cot_inputs, cot_reflect_instruction, i + 1, is_sub_task=True)
        agents.append(f'Reflexion CoT agent {cot_agent.id}, refining final answer, thinking: {thinking4.content}; answer: {answer4.content}')
    
    sub_tasks.append(f'Sub-task 4 output: thinking - {thinking4.content}; answer - {answer4.content}')
    print("Subtask 4: ", sub_tasks[-1])

    # Generate final consolidated answer
    final_answer = await self.make_final_answer(thinking4, answer4, sub_tasks, agents)
    return final_answer
