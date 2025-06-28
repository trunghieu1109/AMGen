async def forward(self, taskInfo):

    from collections import Counter
    
    print("Task Requirement: ", taskInfo)
    
    # Initialize lists to keep track of sub-tasks and agents
    sub_tasks = []
    agents = []

    """
    [Stage 1: Derive Target Output]
    
    [Objective] 
    - Derive the target output by applying defined transformations, operations, or mappings to given inputs under specified conditions or rules.
    
    [Agent Collaborations]
    - Use Self-Consistency Chain-of-Thought (SC_CoT) and Reflexion to ensure consistent and valid derivation of the target output.
    
    [Examples]
    - Implement subtasks to analyze expressions and derive outputs using SC_CoT and Reflexion.
    """
    
    # Sub-task 1: Derive target output using SC_CoT
    cot_sc_instruction = "Sub-task 1: Derive the target output by considering all possible transformations and operations."
    N = self.max_sc
    cot_agents = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N)]
    possible_answers = []
    thinkingmapping = {}
    answermapping = {}
    
    for i in range(N):
        thinking1, answer1 = await cot_agents[i]([taskInfo], cot_sc_instruction, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents[i].id}, deriving target output, thinking: {thinking1.content}; answer: {answer1.content}")
        possible_answers.append(answer1.content)
        thinkingmapping[answer1.content] = thinking1
        answermapping[answer1.content] = answer1
    
    most_common_answer = Counter(possible_answers).most_common(1)[0][0]
    thinking1 = thinkingmapping[most_common_answer]
    answer1 = answermapping[most_common_answer]
    sub_tasks.append(f"Sub-task 1 output: thinking - {thinking1.content}; answer - {answer1.content}")
    
    print("Subtask 1 answer: ", sub_tasks[-1])
    
    """
    [Stage 2: Transform and Integrate Inputs]
    
    [Objective] 
    - Apply one or more defined operations to one or multiple inputs to generate outputs, which may be sequentially ordered or combined into a composite result.
    
    [Agent Collaborations]
    - Use Chain-of-Thought (CoT) and Reflexion to transform and integrate inputs effectively.
    
    [Examples]
    - Implement subtasks to transform inputs and integrate them using CoT and Reflexion.
    """
    
    # Sub-task 2: Transform and integrate inputs with Reflexion
    cot_reflect_instruction = "Sub-task 2: Transform and integrate inputs based on derived outputs from Sub-task 1."
    cot_agent = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    critic_agent = LLMAgentBase(["feedback", "correct"], "Critic Agent", model=self.node_model, temperature=0.0)
    N_max = self.max_round
    
    cot_inputs = [taskInfo, thinking1, answer1]
    
    thinking2, answer2 = await cot_agent(cot_inputs, cot_reflect_instruction, 0, is_sub_task=True)
    agents.append(f"Reflexion CoT agent {cot_agent.id}, transforming and integrating inputs, thinking: {thinking2.content}; answer: {answer2.content}")

    for i in range(N_max):
        feedback, correct = await critic_agent([taskInfo, thinking2, answer2], "Critically evaluate the transformation and integration process.", i, is_sub_task=True)
        agents.append(f"Critic agent {critic_agent.id}, providing feedback, thinking: {feedback.content}; answer: {correct.content}")
        if correct.content == "True":
            break
        
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
    - Use Self-Consistency Chain-of-Thought (SC_CoT) and Reflexion to assess the impact of modifications accurately.
    
    [Examples]
    - Implement subtasks to assess modification impacts using SC_CoT and Reflexion.
    """
    
    # Sub-task 3: Assess modification impact using Debate
    debate_instruction_3 = "Sub-task 3: Assess the impact of modifications based on outputs from previous subtasks."
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
            agents.append(f"Debate agent {agent.id}, round {r}, assessing modification impact, thinking: {thinking3.content}; answer: {answer3.content}")
            all_thinking3[r].append(thinking3)
            all_answer3[r].append(answer3)
            
    final_decision_agent_3 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking3, answer3 = await final_decision_agent_3([taskInfo] + all_thinking3[-1] + all_answer3[-1], "Sub-task 3: Make a final decision on the modification impact.", is_sub_task=True)
    agents.append(f"Final Decision agent on assessing modification impact, thinking: {thinking3.content}; answer: {answer3.content}")
    sub_tasks.append(f"Sub-task 3 output: thinking - {thinking3.content}; answer - {answer3.content}")
    
    print("Subtask 3 answer: ", sub_tasks[-1])
    
    final_answer = await self.make_final_answer(thinking3, answer3, sub_tasks, agents)
    return final_answer