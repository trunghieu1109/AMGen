async def forward_30(self, taskInfo):

    from collections import Counter
    
    print("Task Requirement: ", taskInfo)
    
    sub_tasks = []
    agents = []

    # Stage 1: Apply Transformation
    cot_instruction = "Sub-task 1: Identify the chemical structure of product 1 formed from the reaction of toluene with nitric acid and sulfuric acid, considering the nitration process and expected functional groups."
    cot_agent = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", 
                            model=self.node_model, temperature=0.0)
    thinking1, answer1 = await cot_agent([taskInfo], cot_instruction, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent.id}, identifying product 1 structure, thinking: {thinking1.content}; answer: {answer1.content}")
    sub_tasks.append(f"Sub-task 1 output: thinking - {thinking1.content}; answer - {answer1.content}")
    print("Subtask 1 answer: ", sub_tasks[-1])
    
    # Stage 2: Comprehensive Element Evaluation and Prioritization
    cot_sc_instruction = "Sub-task 2: Determine the chemical structure of product 2 formed from the reaction of product 1 with MnO2 and H2SO4, focusing on oxidation reactions and molecular structure changes."
    N = self.max_sc
    cot_agents = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", 
                              model=self.node_model, temperature=0.5) for _ in range(N)]
    possible_answers = []
    thinkingmapping = {}
    answermapping = {}
    
    for i in range(N):
        thinking2, answer2 = await cot_agents[i]([taskInfo, thinking1, answer1], cot_sc_instruction, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents[i].id}, determining product 2 structure, thinking: {thinking2.content}; answer: {answer2.content}")
        possible_answers.append(answer2.content)
        thinkingmapping[answer2.content] = thinking2
        answermapping[answer2.content] = answer2
    
    # Select the most consistent answer for product 2
    final_answer2 = max(set(possible_answers), key=possible_answers.count)
    thinking2 = thinkingmapping[final_answer2]
    answer2 = answermapping[final_answer2]
    sub_tasks.append(f"Sub-task 2 output: thinking - {thinking2.content}; answer - {answer2.content}")
    print("Subtask 2 answer: ", sub_tasks[-1])
    
    # Stage 3: Final Decision Making
    cot_reflect_instruction = "Sub-task 3: Identify the chemical structure of product 3 formed from the reaction of product 2 with acetone and aqueous sodium hydroxide, focusing on aldol condensation reactions."
    cot_agent = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", 
                            model=self.node_model, temperature=0.0)
    thinking3, answer3 = await cot_agent([taskInfo, thinking2, answer2], cot_reflect_instruction, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent.id}, identifying product 3 structure, thinking: {thinking3.content}; answer: {answer3.content}")
    sub_tasks.append(f"Sub-task 3 output: thinking - {thinking3.content}; answer - {answer3.content}")
    print("Subtask 3 answer: ", sub_tasks[-1])
    
    # Stage 4: Analyze molecular symmetry
    cot_symmetry_instruction = "Sub-task 4: Analyze the molecular symmetry of product 3 to determine its symmetry group, considering symmetry elements and classification."
    cot_agent = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", 
                            model=self.node_model, temperature=0.0)
    thinking4, answer4 = await cot_agent([taskInfo, thinking3, answer3], cot_symmetry_instruction, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent.id}, analyzing symmetry of product 3, thinking: {thinking4.content}; answer: {answer4.content}")
    sub_tasks.append(f"Sub-task 4 output: thinking - {thinking4.content}; answer - {answer4.content}")
    print("Subtask 4 answer: ", sub_tasks[-1])
    
    final_answer = await self.make_final_answer(thinking4, answer4, sub_tasks, agents)
    return final_answer