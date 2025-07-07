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
    - Implementing a transformation operation on input data.
    """
    
    # Sub-task 1: Apply transformation using CoT
    cot_instruction = "Sub-task 1: Apply the specified transformation to the input data."
    cot_agent = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", 
                            model=self.node_model, temperature=0.0)
    thinking1, answer1 = await cot_agent([taskInfo], cot_instruction, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent.id}, applying transformation, thinking: {thinking1.content}; answer: {answer1.content}")
    sub_tasks.append(f"Sub-task 1 output: thinking - {thinking1.content}; answer - {answer1.content}")
    
    print("Subtask 1 answer: ", sub_tasks[-1])
    
    """
    [Stage 2: Assess Modification Impact]
    
    [Objective] 
    - Assess the effect of a specified change or transformation on the state, properties, or measurable outcomes of a target entity or system.
    
    [Agent Collaborations]
    - Self-Consistency Chain-of-Thought (SC_CoT) and Reflexion
    
    [Examples]
    - Evaluating the impact of a transformation on system performance.
    """
    
    # Sub-task 2: Assess impact using Reflexion
    cot_reflect_instruction = "Sub-task 2: Assess the impact of the transformation on the system."
    cot_agent = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", 
                            model=self.node_model, temperature=0.0)
    critic_agent = LLMAgentBase(["feedback", "correct"], "Critic Agent", 
                               model=self.node_model, temperature=0.0)
    N_max = self.max_round
    
    cot_inputs = [taskInfo, thinking1, answer1]
    
    thinking2, answer2 = await cot_agent(cot_inputs, cot_reflect_instruction, 0, is_sub_task=True)
    agents.append(f"Reflexion CoT agent {cot_agent.id}, assessing impact, thinking: {thinking2.content}; answer: {answer2.content}")

    for i in range(N_max):
        feedback, correct = await critic_agent([taskInfo, thinking2, answer2], 
                                       "Critically evaluate the impact assessment and provide its limitations.", 
                                       i, is_sub_task=True)
        agents.append(f"Critic agent {critic_agent.id}, providing feedback, thinking: {feedback.content}; answer: {correct.content}")
        if correct.content == "True":
            break
        
        cot_inputs.extend([thinking2, answer2, feedback])
        thinking2, answer2 = await cot_agent(cot_inputs, cot_reflect_instruction, i + 1, is_sub_task=True)
        agents.append(f"Reflexion CoT agent {cot_agent.id}, refining impact assessment, thinking: {thinking2.content}; answer: {answer2.content}")
    
    sub_tasks.append(f"Sub-task 2 output: thinking - {thinking2.content}; answer - {answer2.content}")
    
    print("Subtask 2 answer: ", sub_tasks[-1])
    
    """
    [Stage 3: Transform and Generate Variants]
    
    [Objective] 
    - Define transformation criteria and generate variant configurations by applying these criteria to input elements, optionally assessing the significance of the resulting variants.
    
    [Agent Collaborations]
    - Chain-of-Thought (CoT) and Debate
    
    [Examples]
    - Generating different configurations of a product based on transformation criteria.
    """
    
    # Sub-task 3: Generate variants using Debate
    debate_instruction_3 = "Sub-task 3: Generate variant configurations based on transformation criteria."
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
            agents.append(f"Debate agent {agent.id}, round {r}, generating variants, thinking: {thinking3.content}; answer: {answer3.content}")
            all_thinking3[r].append(thinking3)
            all_answer3[r].append(answer3)
            
    final_decision_agent_3 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking3, answer3 = await final_decision_agent_3([taskInfo] + all_thinking3[-1] + all_answer3[-1], "Sub-task 3: Make a final decision on the variants.", is_sub_task=True)
    agents.append(f"Final Decision agent on generating variants, thinking: {thinking3.content}; answer: {answer3.content}")
    sub_tasks.append(f"Sub-task 3 output: thinking - {thinking3.content}; answer - {answer3.content}")
    
    print("Subtask 3 answer: ", sub_tasks[-1])
    
    """
    [Stage 4: Select Elements by Criteria Conformity]
    
    [Objective] 
    - Evaluate elements within a collection against defined criteria or conditions and identify those that satisfy or fail to satisfy these criteria.
    
    [Agent Collaborations]
    - Debate and Self-Consistency Chain-of-Thought (SC_CoT)
    
    [Examples]
    - Selecting elements from a dataset that meet specific criteria.
    """
    
    # Sub-task 4: Select elements using SC_CoT
    cot_sc_instruction = "Sub-task 4: Evaluate elements against criteria and select those that conform."
    N = self.max_sc
    cot_agents = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", 
                              model=self.node_model, temperature=0.5) for _ in range(N)]
    possible_answers = []
    thinkingmapping = {}
    answermapping = {}
    
    for i in range(N):
        thinking4, answer4 = await cot_agents[i]([taskInfo, thinking3, answer3], cot_sc_instruction, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents[i].id}, evaluating elements, thinking: {thinking4.content}; answer: {answer4.content}")
        possible_answers.append(answer4.content)
        thinkingmapping[answer4.content] = thinking4
        answermapping[answer4.content] = answer4
    
    most_common_answer = Counter(possible_answers).most_common(1)[0][0]
    thinking4 = thinkingmapping[most_common_answer]
    answer4 = answermapping[most_common_answer]
    sub_tasks.append(f"Sub-task 4 output: thinking - {thinking4.content}; answer - {answer4.content}")
    
    print("Subtask 4 answer: ", sub_tasks[-1])
    
    final_answer = await self.make_final_answer(thinking4, answer4, sub_tasks, agents)
    return final_answer