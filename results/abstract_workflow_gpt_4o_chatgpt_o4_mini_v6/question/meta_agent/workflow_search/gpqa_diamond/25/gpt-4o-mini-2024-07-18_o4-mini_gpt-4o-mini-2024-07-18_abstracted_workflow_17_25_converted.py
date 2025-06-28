async def forward_25(self, taskInfo):
    from collections import Counter
    
    print("Task Requirement: ", taskInfo)
    
    sub_tasks = []
    agents = []
    
    # Stage 1: Evaluate and Select Reactant
    cot_instruction = "Sub-task 1: Evaluate the provided reactant options (A) to determine which one can react with Cyclohexene to form the specified product (8,8-diiodobicyclo[4.2.0]octan-7-one). This involves analyzing the structure and reactivity of each option in relation to the reaction conditions."
    cot_agent = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", 
                            model=self.node_model, temperature=0.0)
    thinking1, answer1 = await cot_agent([taskInfo], cot_instruction, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent.id}, evaluating reactants, thinking: {thinking1.content}; answer: {answer1.content}")
    sub_tasks.append(f"Sub-task 1 output: thinking - {thinking1.content}; answer - {answer1.content}")
    
    print("Subtask 1 answer: ", sub_tasks[-1])
    
    # Stage 2: Analyze and Select Dienes
    cot_instruction_2 = "Sub-task 3: Analyze the provided dienes (B) to determine their reactivity order based on their structural features and known reactivity patterns in organic chemistry. This will help in establishing a ranking from most reactive to least reactive."
    cot_agent_2 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", 
                            model=self.node_model, temperature=0.0)
    thinking2, answer2 = await cot_agent_2([taskInfo], cot_instruction_2, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_2.id}, analyzing dienes, thinking: {thinking2.content}; answer: {answer2.content}")
    sub_tasks.append(f"Sub-task 3 output: thinking - {thinking2.content}; answer - {answer2.content}")
    
    print("Subtask 3 answer: ", sub_tasks[-1])
    
    cot_sc_instruction = "Sub-task 4: Select the correct sequence of the dienes according to their reactivity from the analysis in subtask 3. This will involve comparing the reactivity rankings derived from the analysis."
    N = self.max_sc
    cot_agents = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", 
                              model=self.node_model, temperature=0.5) for _ in range(N)]
    possible_answers = []
    thinkingmapping = {}
    answermapping = {}
    
    for i in range(N):
        thinking3, answer3 = await cot_agents[i]([taskInfo, thinking2, answer2], cot_sc_instruction, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents[i].id}, selecting dienes, thinking: {thinking3.content}; answer: {answer3.content}")
        possible_answers.append(answer3.content)
        thinkingmapping[answer3.content] = thinking3
        answermapping[answer3.content] = answer3
    
    most_common_answer = Counter(possible_answers).most_common(1)[0][0]
    thinking3 = thinkingmapping[most_common_answer]
    answer3 = answermapping[most_common_answer]
    sub_tasks.append(f"Sub-task 4 output: thinking - {thinking3.content}; answer - {answer3.content}")
    
    print("Subtask 4 answer: ", sub_tasks[-1])
    
    final_answer = await self.make_final_answer(thinking3, answer3, sub_tasks, agents)
    return final_answer