async def forward_77(self, taskInfo):
    from collections import Counter
    
    print("Task Requirement: ", taskInfo)
    
    sub_tasks = []
    agents = []
    
    # Stage 1: Analyze and Classify Elements
    
    # Sub-task 1: Analyze the physical setup and given variables
    cot_instruction_1 = (
        "Sub-task 1: Analyze the physical setup and given variables: identify the moving point charge q, its trajectory s(t), "
        "the observation point r, the retarded time tr, the vector d from the retarded position to r, and the velocity v of the charge at time tr. "
        "Clarify the meaning of all symbols and their relationships in the context of electromagnetic potentials."
    )
    cot_agent_1 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    thinking1, answer1 = await cot_agent_1([taskInfo], cot_instruction_1, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_1.id}, analyzing physical setup, thinking: {thinking1.content}; answer: {answer1.content}")
    sub_tasks.append(f"Sub-task 1 output: thinking - {thinking1.content}; answer - {answer1.content}")
    print("Subtask 1 answer: ", sub_tasks[-1])
    
    # Sub-task 2: Classify the types of potentials involved and their dependencies
    cot_sc_instruction_2 = (
        "Sub-task 2: Based on the output from Sub-task 1, classify the scalar potential V and vector potential A, "
        "their dependence on the charge motion and observation point, including the role of retarded time tr, speed of light c, permittivity epsilon_o, and permeability mu_o."
    )
    N = self.max_sc
    cot_agents_2 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N)]
    possible_answers_2 = []
    thinkingmapping_2 = {}
    answermapping_2 = {}
    for i in range(N):
        thinking2, answer2 = await cot_agents_2[i]([taskInfo, thinking1, answer1], cot_sc_instruction_2, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents_2[i].id}, classifying potentials, thinking: {thinking2.content}; answer: {answer2.content}")
        possible_answers_2.append(answer2.content)
        thinkingmapping_2[answer2.content] = thinking2
        answermapping_2[answer2.content] = answer2
    # Choose the most common answer for consistency
    answer2_content = Counter(possible_answers_2).most_common(1)[0][0]
    thinking2_content = thinkingmapping_2[answer2_content].content
    sub_tasks.append(f"Sub-task 2 output: thinking - {thinking2_content}; answer - {answer2_content}")
    print("Subtask 2 answer: ", sub_tasks[-1])
    
    # Stage 2: Derive, Analyze, and Confirm Potentials
    
    # Sub-task 3: Derive or recall the general expressions for scalar and vector potentials
    cot_instruction_3 = (
        "Sub-task 3: Derive or recall the general expressions for the scalar potential V and vector potential A for a moving point charge, "
        "incorporating retarded time and vector d, using Lienard-Wiechert potentials."
    )
    cot_agent_3 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    thinking3, answer3 = await cot_agent_3([taskInfo, thinking2, answermapping_2[answer2_content]], cot_instruction_3, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_3.id}, deriving potentials, thinking: {thinking3.content}; answer: {answer3.content}")
    sub_tasks.append(f"Sub-task 3 output: thinking - {thinking3.content}; answer - {answer3.content}")
    print("Subtask 3 answer: ", sub_tasks[-1])
    
    # Sub-task 4: Analyze and interpret the given four choices of potentials
    cot_instruction_4 = (
        "Sub-task 4: Analyze and interpret the four given choices of potentials, comparing their forms to the derived expressions from Sub-task 3, "
        "focusing on denominators and vector dependencies, to identify which choice correctly represents the potentials at time t and position r."
    )
    cot_agent_4 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    thinking4, answer4 = await cot_agent_4([taskInfo, thinking3, answer3], cot_instruction_4, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_4.id}, analyzing choices, thinking: {thinking4.content}; answer: {answer4.content}")
    sub_tasks.append(f"Sub-task 4 output: thinking - {thinking4.content}; answer - {answer4.content}")
    print("Subtask 4 answer: ", sub_tasks[-1])
    
    # Sub-task 5: Confirm physical consistency of the selected potentials
    cot_reflect_instruction_5 = (
        "Sub-task 5: Confirm the physical consistency of the selected potentials with respect to known constants (c, epsilon_o, mu_o), "
        "vector and scalar forms, and the causality condition t > tr, ensuring the solution satisfies electromagnetic theory."
    )
    cot_agent_5 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    critic_agent_5 = LLMAgentBase(["feedback", "correct"], "Critic Agent", model=self.node_model, temperature=0.0)
    N_max_5 = self.max_round
    cot_inputs_5 = [taskInfo, thinking4, answer4]
    thinking5, answer5 = await cot_agent_5(cot_inputs_5, cot_reflect_instruction_5, 0, is_sub_task=True)
    agents.append(f"Reflexion CoT agent {cot_agent_5.id}, confirming physical consistency, thinking: {thinking5.content}; answer: {answer5.content}")
    for i in range(N_max_5):
        feedback, correct = await critic_agent_5([taskInfo, thinking5, answer5], 
                                               "Review the physical consistency and correctness of the selected potentials.", 
                                               i, is_sub_task=True)
        agents.append(f"Critic agent {critic_agent_5.id}, providing feedback, thinking: {feedback.content}; answer: {correct.content}")
        if correct.content == "True":
            break
        cot_inputs_5.extend([thinking5, answer5, feedback])
        thinking5, answer5 = await cot_agent_5(cot_inputs_5, cot_reflect_instruction_5, i + 1, is_sub_task=True)
        agents.append(f"Reflexion CoT agent {cot_agent_5.id}, refining physical consistency, thinking: {thinking5.content}; answer: {answer5.content}")
    sub_tasks.append(f"Sub-task 5 output: thinking - {thinking5.content}; answer - {answer5.content}")
    print("Subtask 5 answer: ", sub_tasks[-1])
    
    # Final answer synthesis
    final_answer = await self.make_final_answer(thinking5, answer5, sub_tasks, agents)
    return final_answer
