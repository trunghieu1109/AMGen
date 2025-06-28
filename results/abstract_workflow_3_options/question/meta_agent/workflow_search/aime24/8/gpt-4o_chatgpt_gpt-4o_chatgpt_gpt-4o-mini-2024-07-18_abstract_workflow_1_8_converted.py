async def forward(self, taskInfo):
    from collections import Counter
    
    sub_tasks = []
    agents = []

    # Stage 1: Condition Identification and Decomposition
    
    # Sub-task 1: Identify base cases for the game
    cot_instruction_1 = "Sub-task 1: Identify the base cases for the game, i.e., the smallest values of n for which the outcome is known (winning or losing position)."
    cot_agent_1 = LLMAgentBase(['thinking', 'answer'], 'Chain-of-Thought Agent', model=self.node_model, temperature=0.0)
    thinking1, answer1 = cot_agent_1([taskInfo], cot_instruction_1, is_sub_task=True)
    agents.append(f'CoT agent {cot_agent_1.id}, identifying base cases, thinking: {thinking1.content}; answer: {answer1.content}')
    sub_tasks.append(f'Sub-task 1 output: thinking - {thinking1.content}; answer - {answer1.content}')
    print("Step 1: ", sub_tasks[-1])

    # Sub-task 2: Determine recursive conditions
    cot_instruction_2 = "Sub-task 2: Based on the output from Sub-task 1, determine the recursive conditions or rules that define winning and losing positions."
    cot_agent_2 = LLMAgentBase(['thinking', 'answer'], 'Chain-of-Thought Agent', model=self.node_model, temperature=0.0)
    thinking2, answer2 = cot_agent_2([taskInfo, thinking1, answer1], cot_instruction_2, is_sub_task=True)
    agents.append(f'CoT agent {cot_agent_2.id}, determining recursive conditions, thinking: {thinking2.content}; answer: {answer2.content}')
    sub_tasks.append(f'Sub-task 2 output: thinking - {thinking2.content}; answer - {answer2.content}')
    print("Step 2: ", sub_tasks[-1])

    # Sub-task 3: Identify strategy for Bob
    cot_instruction_3 = "Sub-task 3: Identify the pattern or strategy that Bob can use to ensure a win, based on the recursive conditions."
    cot_agent_3 = LLMAgentBase(['thinking', 'answer'], 'Chain-of-Thought Agent', model=self.node_model, temperature=0.0)
    thinking3, answer3 = cot_agent_3([taskInfo, thinking2, answer2], cot_instruction_3, is_sub_task=True)
    agents.append(f'CoT agent {cot_agent_3.id}, identifying strategy for Bob, thinking: {thinking3.content}; answer: {answer3.content}')
    sub_tasks.append(f'Sub-task 3 output: thinking - {thinking3.content}; answer - {answer3.content}')
    print("Step 3: ", sub_tasks[-1])

    # Stage 2: Inference and Synthesis
    
    # Sub-task 4: Count the number of integers for which Bob can guarantee a win
    cot_instruction_4 = "Sub-task 4: Count the number of positive integers n less than or equal to 2024 for which Bob can guarantee a win."
    cot_agent_4 = LLMAgentBase(['thinking', 'answer'], 'Chain-of-Thought Agent', model=self.node_model, temperature=0.0)
    thinking4, answer4 = cot_agent_4([taskInfo, thinking3, answer3], cot_instruction_4, is_sub_task=True)
    agents.append(f'CoT agent {cot_agent_4.id}, counting winning positions for Bob, thinking: {thinking4.content}; answer: {answer4.content}')
    sub_tasks.append(f'Sub-task 4 output: thinking - {thinking4.content}; answer - {answer4.content}')
    print("Step 4: ", sub_tasks[-1])

    # Sub-task 5: Synthesize results for final answer
    cot_reflect_instruction_5 = "Sub-task 5: Synthesize the results to provide a final answer to the query, ensuring all conditions and strategies are considered."
    cot_agent_5 = LLMAgentBase(['thinking', 'answer'], 'Chain-of-Thought Agent', model=self.node_model, temperature=0.0)
    critic_agent_5 = LLMAgentBase(['feedback', 'correct'], 'Critic Agent', model=self.node_model, temperature=0.0)
    N_max_5 = self.max_round
    
    cot_inputs_5 = [taskInfo, thinking4, answer4]
    thinking5, answer5 = cot_agent_5(cot_inputs_5, cot_reflect_instruction_5, 0, is_sub_task=True)
    agents.append(f'Reflexion CoT agent {cot_agent_5.id}, synthesizing final answer, thinking: {thinking5.content}; answer: {answer5.content}')

    for i in range(N_max_5):
        feedback, correct = critic_agent_5([taskInfo, thinking5, answer5], "please review the final answer synthesis and provide feedback about its limitations.", i, is_sub_task=True)
        agents.append(f'Critic agent {critic_agent_5.id}, providing feedback, thinking: {feedback.content}; answer: {correct.content}')
        if correct.content == 'True':
            break
        cot_inputs_5.extend([thinking5, answer5, feedback])
        thinking5, answer5 = cot_agent_5(cot_inputs_5, cot_reflect_instruction_5, i + 1, is_sub_task=True)
        agents.append(f'Reflexion CoT agent {cot_agent_5.id}, refining final answer, thinking: {thinking5.content}; answer: {answer5.content}')
    sub_tasks.append(f'Sub-task 5 output: thinking - {thinking5.content}; answer - {answer5.content}')
    print("Step 5: ", sub_tasks[-1])

    final_answer = await self.make_final_answer(thinking5, answer5, sub_tasks, agents)
    return final_answer