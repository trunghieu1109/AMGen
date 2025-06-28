async def forward(self, taskInfo):
    from collections import Counter
    
    print("Task Requirement: ", taskInfo)
    
    # Initialize lists to keep track of sub-tasks and agents
    sub_tasks = []
    agents = []

    """
    [Stage 1: Apply Transformation]
    
    [Objective] 
    - Apply a specified operation or transformation to an input to produce a corresponding output.
    
    [Agent Collaborations]
    - Chain-of-Thought (CoT) and Self-Consistency Chain-of-Thought (SC_CoT)
    
    [Examples]
    - Analyze first expression/data component with self-consistency
    """
    
    # Sub-task 1: Apply transformation using CoT
    cot_instruction = "Sub-task 1: Apply transformation to the input data with context from [taskInfo]"
    cot_agent = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", 
                            model=self.node_model, temperature=0.0)
    thinking1, answer1 = await cot_agent([taskInfo], cot_instruction, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent.id}, applying transformation, thinking: {thinking1.content}; answer: {answer1.content}")
    sub_tasks.append(f"Sub-task 1 output: thinking - {thinking1.content}; answer - {answer1.content}")
    
    print("Subtask 1 answer: ", sub_tasks[-1])
    
    """
    [Stage 2: Transform and Integrate Inputs]
    
    [Objective] 
    - Apply one or more defined operations to one or multiple inputs to generate one or more outputs, which may be sequentially ordered or combined into a composite result.
    
    [Agent Collaborations]
    - Chain-of-Thought (CoT) and Reflexion
    
    [Examples]
    - Calculate intermediate output with reflexion
    """
    
    # Sub-task 2: Transform and integrate inputs using Reflexion
    cot_reflect_instruction = "Sub-task 2: Based on Sub-task 1 outputs, transform and integrate inputs"
    cot_agent = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", 
                            model=self.node_model, temperature=0.0)
    critic_agent = LLMAgentBase(["feedback", "correct"], "Critic Agent", 
                               model=self.node_model, temperature=0.0)
    N_max = self.max_round
    
    # Input aggregation from previous stages
    cot_inputs = [taskInfo, thinking1, answer1]
    
    # Generate initial intermediate computation
    thinking2, answer2 = await cot_agent(cot_inputs, cot_reflect_instruction, 0, is_sub_task=True)
    agents.append(f"Reflexion CoT agent {cot_agent.id}, transforming and integrating inputs, thinking: {thinking2.content}; answer: {answer2.content}")

    # Iterative refinement through critic feedback
    for i in range(N_max):
        feedback, correct = await critic_agent([taskInfo, thinking2, answer2], 
                                       "Critically evaluate the transformation and integration, and provide its limitations.", 
                                       i, is_sub_task=True)
        agents.append(f"Critic agent {critic_agent.id}, providing feedback, thinking: {feedback.content}; answer: {correct.content}")
        if correct.content == "True":
            break
        
        # Incorporate feedback for next iteration
        cot_inputs.extend([thinking2, answer2, feedback])
        thinking2, answer2 = await cot_agent(cot_inputs, cot_reflect_instruction, i + 1, is_sub_task=True)
        agents.append(f"Reflexion CoT agent {cot_agent.id}, refining transformation and integration, thinking: {thinking2.content}; answer: {answer2.content}")
    
    sub_tasks.append(f"Sub-task 2 output: thinking - {thinking2.content}; answer - {answer2.content}")
    
    print("Subtask 2 answer: ", sub_tasks[-1])
    
    """
    [Stage 3: Assess Modification Impact]
    
    [Objective] 
    - Assess the effect of a specified change or transformation on the state, properties, or measurable outcomes of a target entity or system.
    
    [Agent Collaborations]
    - Self-Consistency Chain-of-Thought (SC_CoT) and Reflexion
    
    [Examples]
    - Assess modification impact using SC_CoT
    """
    
    # Sub-task 3: Assess modification impact using SC_CoT
    cot_sc_instruction = "Sub-task 3: Assess the impact of modifications based on previous outputs"
    N = self.max_sc
    cot_agents = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", 
                              model=self.node_model, temperature=0.5) for _ in range(N)]
    possible_answers = []
    thinkingmapping = {}
    answermapping = {}
    
    for i in range(N):
        thinking3, answer3 = await cot_agents[i]([taskInfo, thinking2, answer2], cot_sc_instruction, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents[i].id}, assessing modification impact, thinking: {thinking3.content}; answer: {answer3.content}")
        possible_answers.append(answer3.content)
        thinkingmapping[answer3.content] = thinking3
        answermapping[answer3.content] = answer3
    
    # Determine the most common answer
    most_common_answer = Counter(possible_answers).most_common(1)[0][0]
    thinking3 = thinkingmapping[most_common_answer]
    answer3 = answermapping[most_common_answer]
    sub_tasks.append(f"Sub-task 3 output: thinking - {thinking3.content}; answer - {answer3.content}")
    
    print("Subtask 3 answer: ", sub_tasks[-1])
    
    """
    [Stage 4: Transform and Generate Variants]
    
    [Objective] 
    - Define transformation criteria and generate variant configurations by applying these criteria to input elements, optionally assessing the significance of the resulting variants.
    
    [Agent Collaborations]
    - Chain-of-Thought (CoT) and Debate
    
    [Examples]
    - Generate variants using Debate
    """
    
    # Sub-task 4: Transform and generate variants using Debate
    debate_instruction_4 = "Sub-task 4: Generate variants based on transformation criteria"
    debate_agents_4 = [LLMAgentBase(["thinking", "answer"], "Debate Agent", model=self.node_model, role=role, temperature=0.5) for role in self.debate_role]
    N_max_4 = self.max_round
    all_thinking4 = [[] for _ in range(N_max_4)]
    all_answer4 = [[] for _ in range(N_max_4)]
    
    for r in range(N_max_4):
        for i, agent in enumerate(debate_agents_4):
            input_infos_4 = [taskInfo, thinking3, answer3]
            if r > 0:
                input_infos_4.extend(all_thinking4[r-1])
            thinking4, answer4 = await agent(input_infos_4, debate_instruction_4, is_sub_task=True)
            agents.append(f"Debate agent {agent.id}, round {r}, generating variants, thinking: {thinking4.content}; answer: {answer4.content}")
            all_thinking4[r].append(thinking4)
            all_answer4[r].append(answer4)
    
    final_decision_agent_4 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking4, answer4 = await final_decision_agent_4([taskInfo] + all_thinking4[-1] + all_answer4[-1], "Sub-task 4: Make a final decision on the variants.", is_sub_task=True)
    agents.append(f"Final Decision agent on generating variants, thinking: {thinking4.content}; answer: {answer4.content}")
    sub_tasks.append(f"Sub-task 4 output: thinking - {thinking4.content}; answer - {answer4.content}")
    
    print("Subtask 4 answer: ", sub_tasks[-1])
    
    final_answer = await self.make_final_answer(thinking4, answer4, sub_tasks, agents)
    return final_answer