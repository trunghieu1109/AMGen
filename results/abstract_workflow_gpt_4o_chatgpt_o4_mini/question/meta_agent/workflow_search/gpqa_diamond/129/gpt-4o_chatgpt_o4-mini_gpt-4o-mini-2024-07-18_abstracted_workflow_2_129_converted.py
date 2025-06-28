async def forward_129(self, taskInfo):
    from collections import Counter
    
    print("Task Requirement: ", taskInfo)
    
    sub_tasks = []
    agents = []

    # Stage 1: Calculate Observed Wavelength and Optical Range
    
    # Sub-task 1: Calculate observed wavelength using CoT
    cot_instruction_1 = "Sub-task 1: Calculate the observed wavelength of the Lyman alpha line for a given redshift using the formula: observed_wavelength = (1 + redshift) * rest_wavelength."
    cot_agent_1 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", 
                            model=self.node_model, temperature=0.0)
    thinking1, answer1 = await cot_agent_1([taskInfo], cot_instruction_1, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_1.id}, calculating observed wavelength, thinking: {thinking1.content}; answer: {answer1.content}")
    sub_tasks.append(f"Sub-task 1 output: thinking - {thinking1.content}; answer - {answer1.content}")
    
    print("Subtask 1 answer: ", sub_tasks[-1])

    # Sub-task 2: Determine optical range using CoT
    cot_instruction_2 = "Sub-task 2: Determine the range of wavelengths that optical ground-based telescopes can detect, typically around 3800 to 7500 Angstroms."
    cot_agent_2 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", 
                            model=self.node_model, temperature=0.0)
    thinking2, answer2 = await cot_agent_2([taskInfo], cot_instruction_2, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_2.id}, determining optical range, thinking: {thinking2.content}; answer: {answer2.content}")
    sub_tasks.append(f"Sub-task 2 output: thinking - {thinking2.content}; answer - {answer2.content}")
    
    print("Subtask 2 answer: ", sub_tasks[-1])

    # Stage 2: Set Up and Solve Inequality
    
    # Sub-task 3: Set up inequality using Reflexion
    cot_reflect_instruction_3 = "Sub-task 3: Set up an inequality to find the minimum redshift such that the observed Lyman alpha wavelength falls within the optical range: (1 + redshift) * 1216 Angstrom >= 3800 Angstrom."
    cot_agent_3 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", 
                            model=self.node_model, temperature=0.0)
    critic_agent_3 = LLMAgentBase(["feedback", "correct"], "Critic Agent", 
                               model=self.node_model, temperature=0.0)
    N_max_3 = self.max_round
    
    cot_inputs_3 = [taskInfo, thinking1, answer1, thinking2, answer2]
    
    thinking3, answer3 = await cot_agent_3(cot_inputs_3, cot_reflect_instruction_3, 0, is_sub_task=True)
    agents.append(f"Reflexion CoT agent {cot_agent_3.id}, setting up inequality, thinking: {thinking3.content}; answer: {answer3.content}")

    for i in range(N_max_3):
        feedback, correct = await critic_agent_3([taskInfo, thinking3, answer3], 
                                       "Review inequality setup and provide feedback.", 
                                       i, is_sub_task=True)
        agents.append(f"Critic agent {critic_agent_3.id}, providing feedback, thinking: {feedback.content}; answer: {correct.content}")
        if correct.content == "True":
            break
        
        cot_inputs_3.extend([thinking3, answer3, feedback])
        thinking3, answer3 = await cot_agent_3(cot_inputs_3, cot_reflect_instruction_3, i + 1, is_sub_task=True)
        agents.append(f"Reflexion CoT agent {cot_agent_3.id}, refining inequality setup, thinking: {thinking3.content}; answer: {answer3.content}")
    
    sub_tasks.append(f"Sub-task 3 output: thinking - {thinking3.content}; answer - {answer3.content}")
    
    print("Subtask 3 answer: ", sub_tasks[-1])

    # Sub-task 4: Solve inequality using CoT
    cot_instruction_4 = "Sub-task 4: Solve the inequality from subtask_3 to find the lower limit on the redshift."
    cot_agent_4 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", 
                            model=self.node_model, temperature=0.0)
    thinking4, answer4 = await cot_agent_4([taskInfo, thinking3, answer3], cot_instruction_4, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_4.id}, solving inequality, thinking: {thinking4.content}; answer: {answer4.content}")
    sub_tasks.append(f"Sub-task 4 output: thinking - {thinking4.content}; answer - {answer4.content}")
    
    print("Subtask 4 answer: ", sub_tasks[-1])

    final_answer = await self.make_final_answer(thinking4, answer4, sub_tasks, agents)
    return final_answer