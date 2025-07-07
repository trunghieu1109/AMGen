async def forward(self, taskInfo):
    from collections import Counter
    
    print("Task Requirement: ", taskInfo)
    
    # Initialize lists to keep track of sub-tasks and agents
    sub_tasks = []
    agents = []

    """
    [Stage 1: Extract Defining Features]
    
    [Objective] 
    - Analyze an input entity or dataset to identify, isolate, and characterize its essential components, attributes, and relationships that define its fundamental structure or nature.
    
    [Agent Collaborations]
    - Use Chain-of-Thought and Debate to thoroughly analyze and discuss the defining features.
    
    [Examples]
    - Extract key features using CoT.
    - Debate the significance of identified features.
    """
    
    # Sub-task 1: Extract key features using CoT
    cot_instruction = "Sub-task 1: Analyze the input to identify key features and attributes."
    cot_agent = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", 
                            model=self.node_model, temperature=0.0)
    thinking1, answer1 = await cot_agent([taskInfo], cot_instruction, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent.id}, analyzing key features, thinking: {thinking1.content}; answer: {answer1.content}")
    sub_tasks.append(f"Sub-task 1 output: thinking - {thinking1.content}; answer - {answer1.content}")
    
    print("Subtask 1 answer: ", sub_tasks[-1])
    
    """
    [Stage 2: Transform and Generate Variants]
    
    [Objective] 
    - Define transformation criteria and generate variant configurations by applying these criteria to input elements, optionally assessing the significance of the resulting variants.
    
    [Agent Collaborations]
    - Use Chain-of-Thought and Debate to explore transformation possibilities and discuss their implications.
    
    [Examples]
    - Generate variants using CoT.
    - Debate the potential impact of each variant.
    """
    
    # Sub-task 2: Generate variants using CoT
    cot_instruction = "Sub-task 2: Define transformation criteria and generate variants."
    cot_agent = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", 
                            model=self.node_model, temperature=0.0)
    thinking2, answer2 = await cot_agent([taskInfo, thinking1, answer1], cot_instruction, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent.id}, generating variants, thinking: {thinking2.content}; answer: {answer2.content}")
    sub_tasks.append(f"Sub-task 2 output: thinking - {thinking2.content}; answer - {answer2.content}")
    
    print("Subtask 2 answer: ", sub_tasks[-1])
    
    """
    [Stage 3: Assess Modification Impact]
    
    [Objective] 
    - Assess the effect of a specified change or transformation on the state, properties, or measurable outcomes of a target entity or system.
    
    [Agent Collaborations]
    - Use Self-Consistency Chain-of-Thought and Reflexion to ensure consistent assessment and iterative refinement.
    
    [Examples]
    - Assess impact using SC_CoT.
    - Refine assessment through Reflexion.
    """
    
    # Sub-task 3: Assess impact using SC_CoT
    cot_sc_instruction = "Sub-task 3: Assess the impact of transformations on the system."
    N = self.max_sc
    cot_agents = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", 
                              model=self.node_model, temperature=0.5) for _ in range(N)]
    possible_answers = []
    thinkingmapping = {}
    answermapping = {}
    
    for i in range(N):
        thinking3, answer3 = await cot_agents[i]([taskInfo, thinking2, answer2], cot_sc_instruction, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents[i].id}, assessing impact, thinking: {thinking3.content}; answer: {answer3.content}")
        possible_answers.append(answer3.content)
        thinkingmapping[answer3.content] = thinking3
        answermapping[answer3.content] = answer3
    
    most_common_answer = Counter(possible_answers).most_common(1)[0][0]
    thinking3 = thinkingmapping[most_common_answer]
    answer3 = answermapping[most_common_answer]
    sub_tasks.append(f"Sub-task 3 output: thinking - {thinking3.content}; answer - {answer3.content}")
    
    print("Subtask 3 answer: ", sub_tasks[-1])
    
    # Final Answer
    final_answer = await self.make_final_answer(thinking3, answer3, sub_tasks, agents)
    return final_answer
}