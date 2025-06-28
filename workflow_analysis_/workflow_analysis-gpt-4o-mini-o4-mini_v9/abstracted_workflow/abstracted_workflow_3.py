async def forward(self, taskInfo):

    from collections import Counter
    
    print("Task Requirement: ", taskInfo)
    
    # Initialize lists to keep track of sub-tasks and agents
    sub_tasks = []
    agents = []

    """
    [Stage 1: Extract Defining Features]
    
    [Objective] 
    - Analyze the input to identify and isolate its essential components, attributes, and relationships.
    
    [Agent Collaborations]
    - Chain-of-Thought (CoT) and Debate
    
    [Examples]
    - Extracting key features from the input data.
    """
    
    # Sub-task 1: Extract defining features using CoT
    cot_instruction = "Sub-task 1: Analyze the input to identify its essential components and relationships."
    cot_agent = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", 
                            model=self.node_model, temperature=0.0)
    thinking1, answer1 = await cot_agent([taskInfo], cot_instruction, is_sub_task=True)
    agents.append(f"CoT agent {{cot_agent.id}}, analyzing input, thinking: {{thinking1.content}}; answer: {{answer1.content}}")
    sub_tasks.append(f"Sub-task 1 output: thinking - {{thinking1.content}}; answer - {{answer1.content}}")
    
    print("Subtask 1 answer: ", sub_tasks[-1])
    
    """
    [Stage 2: Analyze, Classify, Transform, and Generate Variants]
    
    [Objective] 
    - Analyze elements to identify and classify their defining attributes and then apply transformations to generate variant configurations.
    
    [Agent Collaborations]
    - Chain-of-Thought (CoT), Self-Consistency Chain-of-Thought (SC_CoT), and Debate
    
    [Examples]
    - Classifying elements and generating variants.
    """
    
    # Sub-task 2: Generate variants using SC_CoT
    cot_sc_instruction = "Sub-task 2: Based on the output from Sub-task 1, generate variant configurations."
    N = self.max_sc
    cot_agents = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", 
                              model=self.node_model, temperature=0.5) for _ in range(N)]
    possible_answers = []
    thinkingmapping = {}
    answermapping = {}
    
    for i in range(N):
        thinking2, answer2 = await cot_agents[i]([taskInfo, thinking1, answer1], cot_sc_instruction, is_sub_task=True)
        agents.append(f"CoT-SC agent {{cot_agents[i].id}}, generating variants, thinking: {{thinking2.content}}; answer: {{answer2.content}}")
        possible_answers.append(answer2.content)
        thinkingmapping[answer2.content] = thinking2
        answermapping[answer2.content] = answer2
    
    most_common_answer = Counter(possible_answers).most_common(1)[0][0]
    thinking2 = thinkingmapping[most_common_answer]
    answer2 = answermapping[most_common_answer]
    sub_tasks.append(f"Sub-task 2 output: thinking - {{thinking2.content}}; answer - {{answer2.content}}")
    
    print("Subtask 2 answer: ", sub_tasks[-1])
    
    """
    [Stage 3: Evaluate and Prioritize Elements]
    
    [Objective] 
    - Evaluate the collection of elements against specified criteria to identify, select, and rank those that best meet the conditions.
    
    [Agent Collaborations]
    - Debate and Chain-of-Thought (CoT)
    
    [Examples]
    - Evaluating and prioritizing elements based on criteria.
    """
    
    # Sub-task 3: Evaluate and prioritize using Debate
    debate_instruction_3 = "Sub-task 3: Evaluate and prioritize elements based on criteria."
    debate_agents_3 = [LLMAgentBase(["thinking", "answer"], "Debate Agent", model=self.node_model, role=role, temperature=0.5) for role in self.debate_role]
    N_max_3 = self.max_round
    all_thinking3 = [[] for _ in range(N_max_3)]
    all_answer3 = [[] for _ in range(N_max_3)]
    
    for r in range(N_max_3):
        for i, agent in enumerate(debate_agents_3):
            input_infos_3 = [taskInfo, thinking2, answer2]
            if r > 0:
                input_infos_3.extend(all_thinking3[r-1])
            thinking3, answer3 = await agent(input_infos_3, debate_instruction_3, is_sub_task=True)
            agents.append(f"Debate agent {{agent.id}}, round {{r}}, evaluating elements, thinking: {{thinking3.content}}; answer: {{answer3.content}}")
            all_thinking3[r].append(thinking3)
            all_answer3[r].append(answer3)
            
    final_decision_agent_3 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking3, answer3 = await final_decision_agent_3([taskInfo] + all_thinking3[-1] + all_answer3[-1], "Sub-task 3: Make a final decision on the prioritized elements.", is_sub_task=True)
    agents.append(f"Final Decision agent on prioritizing elements, thinking: {{thinking3.content}}; answer: {{answer3.content}}")
    sub_tasks.append(f"Sub-task 3 output: thinking - {{thinking3.content}}; answer - {{answer3.content}}")
    
    print("Subtask 3 answer: ", sub_tasks[-1])
    
    final_answer = await self.make_final_answer(thinking3, answer3, sub_tasks, agents)
    return final_answer
