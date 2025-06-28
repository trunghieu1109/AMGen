async def forward_76(self, taskInfo):

    from collections import Counter
    
    print("Task Requirement: ", taskInfo)
    
    sub_tasks = []
    agents = []

    # Stage 1: Identify starting materials and conditions
    cot_instruction_1 = "Sub-task 1: Identify the structure of the starting materials: (((3-methylbut-2-en-1-yl)oxy)methyl)benzene and the conditions (1. BuLi, 2. H+) for the first reaction."
    cot_agent_1 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", 
                            model=self.node_model, temperature=0.0)
    thinking1, answer1 = await cot_agent_1([taskInfo], cot_instruction_1, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_1.id}, identifying starting materials and conditions, thinking: {thinking1.content}; answer: {answer1.content}")
    sub_tasks.append(f"Sub-task 1 output: thinking - {thinking1.content}; answer - {answer1.content}")
    print("Subtask 1 answer: ", sub_tasks[-1])
    
    # Stage 1: Determine expected product A
    cot_instruction_2 = "Sub-task 2: Determine the expected product A from the rearrangement of the starting material under the specified conditions." 
    N = self.max_sc
    cot_agents_2 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", 
                              model=self.node_model, temperature=0.5) for _ in range(N)]
    possible_answers_A = []
    thinkingmapping_A = {}
    answermapping_A = {}
    
    for i in range(N):
        thinking2, answer2 = await cot_agents_2[i]([taskInfo, thinking1, answer1], cot_instruction_2, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents_2[i].id}, determining expected product A, thinking: {thinking2.content}; answer: {answer2.content}")
        possible_answers_A.append(answer2.content)
        thinkingmapping_A[answer2.content] = thinking2
        answermapping_A[answer2.content] = answer2
    sub_tasks.append(f"Sub-task 2 output: thinking - {thinking2.content}; answer - {answer2.content}")
    print("Subtask 2 answer: ", sub_tasks[-1])
    
    # Stage 2: Analyze second reaction
    cot_instruction_3 = "Sub-task 3: Analyze the second reaction involving 3,4,5,7,8,9-hexamethyl-1,11-dimethylene-2,6,10,11,11a,11b-hexahydro-1H-benzo[cd]indeno[7,1-gh]azulene and the application of heat to predict the product B." 
    cot_agent_3 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", 
                            model=self.node_model, temperature=0.0)
    thinking3, answer3 = await cot_agent_3([taskInfo], cot_instruction_3, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_3.id}, analyzing second reaction, thinking: {thinking3.content}; answer: {answer3.content}")
    sub_tasks.append(f"Sub-task 3 output: thinking - {thinking3.content}; answer - {answer3.content}")
    print("Subtask 3 answer: ", sub_tasks[-1])
    
    # Stage 2: Compare predicted products A and B
    cot_instruction_4 = "Sub-task 4: Compare the predicted products A and B with the provided choices to select the correct pair of products." 
    N_max = self.max_round
    cot_agent_4 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", 
                            model=self.node_model, temperature=0.0)
    critic_agent_4 = LLMAgentBase(["feedback", "correct"], "Critic Agent", 
                               model=self.node_model, temperature=0.0)
    cot_inputs_4 = [taskInfo, thinking2, answer2, thinking3, answer3]
    thinking4, answer4 = await cot_agent_4(cot_inputs_4, cot_instruction_4, 0, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_4.id}, comparing products A and B, thinking: {thinking4.content}; answer: {answer4.content}")
    
    for i in range(N_max):
        feedback, correct = await critic_agent_4([taskInfo, thinking4, answer4], 
                                       "Critically evaluate the comparison of products A and B and provide its limitations.", 
                                       i, is_sub_task=True)
        agents.append(f"Critic agent {critic_agent_4.id}, providing feedback, thinking: {feedback.content}; answer: {correct.content}")
        if correct.content == "True":
            break
        cot_inputs_4.extend([thinking4, answer4, feedback])
        thinking4, answer4 = await cot_agent_4(cot_inputs_4, cot_instruction_4, i + 1, is_sub_task=True)
        agents.append(f"CoT agent {cot_agent_4.id}, refining comparison of products, thinking: {thinking4.content}; answer: {answer4.content}")
    sub_tasks.append(f"Sub-task 4 output: thinking - {thinking4.content}; answer - {answer4.content}")
    print("Subtask 4 answer: ", sub_tasks[-1])
    
    final_answer = await self.make_final_answer(thinking4, answer4, sub_tasks, agents)
    return final_answer