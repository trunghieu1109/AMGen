async def forward_8(self, taskInfo):

    from collections import Counter
    
    print("Task Requirement: ", taskInfo)
    
    sub_tasks = []
    agents = []

    # Stage 1: Identify symmetry properties
    cot_instruction = "Sub-task 1: Identify the symmetry properties of each molecule listed in the query, focusing on determining if they exhibit C3h symmetry."
    cot_agent = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", 
                            model=self.node_model, temperature=0.0)
    thinking1, answer1 = await cot_agent([taskInfo], cot_instruction, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent.id}, identifying symmetry properties, thinking: {thinking1.content}; answer: {answer1.content}")
    sub_tasks.append(f"Sub-task 1 output: thinking - {thinking1.content}; answer - {answer1.content}")
    print("Subtask 1 answer: ", sub_tasks[-1])
    
    # Stage 2: Confirm C3h symmetry
    cot_reflect_instruction = "Sub-task 3: Evaluate the molecular structures of the identified candidates from Sub-task 2 to confirm their C3h symmetry through visual or computational methods."
    cot_agent = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", 
                            model=self.node_model, temperature=0.0)
    critic_agent = LLMAgentBase(["feedback", "correct"], "Critic Agent", 
                               model=self.node_model, temperature=0.0)
    N_max = self.max_round
    
    cot_inputs = [taskInfo, thinking1, answer1]
    thinking2, answer2 = await cot_agent(cot_inputs, cot_reflect_instruction, 0, is_sub_task=True)
    agents.append(f"Reflexion CoT agent {cot_agent.id}, evaluating symmetry confirmation, thinking: {thinking2.content}; answer: {answer2.content}")

    for i in range(N_max):
        feedback, correct = await critic_agent([taskInfo, thinking2, answer2], 
                                       "Critically evaluate the symmetry confirmation and provide its limitations.", 
                                       i, is_sub_task=True)
        agents.append(f"Critic agent {critic_agent.id}, providing feedback, thinking: {feedback.content}; answer: {correct.content}")
        if correct.content == "True":
            break
        cot_inputs.extend([thinking2, answer2, feedback])
        thinking2, answer2 = await cot_agent(cot_inputs, cot_reflect_instruction, i + 1, is_sub_task=True)
        agents.append(f"Reflexion CoT agent {cot_agent.id}, refining symmetry evaluation, thinking: {thinking2.content}; answer: {answer2.content}")
    sub_tasks.append(f"Sub-task 3 output: thinking - {thinking2.content}; answer - {answer2.content}")
    print("Subtask 3 answer: ", sub_tasks[-1])
    
    # Final answer generation
    final_answer = await self.make_final_answer(thinking2, answer2, sub_tasks, agents)
    return final_answer