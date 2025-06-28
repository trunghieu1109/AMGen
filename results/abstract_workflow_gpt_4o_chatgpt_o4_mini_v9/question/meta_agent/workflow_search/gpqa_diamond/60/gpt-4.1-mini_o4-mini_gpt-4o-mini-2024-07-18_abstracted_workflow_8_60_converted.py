async def forward_60(self, taskInfo):
    from collections import Counter
    
    print("Task Requirement: ", taskInfo)
    
    sub_tasks = []
    agents = []
    
    # Stage 1: Determine structures stepwise using Chain-of-Thought and Self-Consistency CoT
    
    # Sub-task 1: Determine product 1 structure after nitration of benzene
    cot_instruction_1 = (
        "Sub-task 1: Determine the chemical structure of product 1 formed by treating benzene with HNO3 and H2SO4, "
        "using knowledge of electrophilic aromatic substitution (nitration) on benzene."
    )
    cot_agent_1 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    thinking1, answer1 = await cot_agent_1([taskInfo], cot_instruction_1, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_1.id}, determining nitration product, thinking: {thinking1.content}; answer: {answer1.content}")
    sub_tasks.append(f"Sub-task 1 output: thinking - {thinking1.content}; answer - {answer1.content}")
    print("Step 1: ", sub_tasks[-1])
    
    # Sub-task 2: Determine product 2 structure after bromination of product 1 with Br2 and iron powder
    cot_sc_instruction_2 = (
        "Sub-task 2: Based on product 1, identify the structure of product 2 formed by treating product 1 with Br2 and iron powder, "
        "considering the directing effects of substituents on the aromatic ring and bromination conditions."
    )
    N_sc = self.max_sc
    cot_agents_2 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N_sc)]
    possible_answers_2 = []
    thinkingmapping_2 = {}
    answermapping_2 = {}
    for i in range(N_sc):
        thinking2, answer2 = await cot_agents_2[i]([taskInfo, thinking1, answer1], cot_sc_instruction_2, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents_2[i].id}, bromination scenarios, thinking: {thinking2.content}; answer: {answer2.content}")
        possible_answers_2.append(answer2.content)
        thinkingmapping_2[answer2.content] = thinking2
        answermapping_2[answer2.content] = answer2
    
    # Choose most consistent answer by frequency
    answer2_counter = Counter(possible_answers_2)
    answer2_final = answer2_counter.most_common(1)[0][0]
    thinking2_final = thinkingmapping_2[answer2_final]
    sub_tasks.append(f"Sub-task 2 output: thinking - {thinking2_final.content}; answer - {answer2_final}")
    print("Step 2: ", sub_tasks[-1])
    
    # Sub-task 3: Determine product 3 structure after catalytic hydrogenation of product 2
    cot_instruction_3 = (
        "Sub-task 3: Determine the structure of product 3 formed by catalytic hydrogenation (Pd/C under H2) of product 2, "
        "focusing on which functional groups or substituents are reduced or remain intact."
    )
    cot_agent_3 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    thinking3, answer3 = await cot_agent_3([taskInfo, thinking1, answer1, thinking2_final, answermapping_2[answer2_final]], cot_instruction_3, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_3.id}, hydrogenation analysis, thinking: {thinking3.content}; answer: {answer3.content}")
    sub_tasks.append(f"Sub-task 3 output: thinking - {thinking3.content}; answer - {answer3.content}")
    print("Step 3: ", sub_tasks[-1])
    
    # Stage 2: Further transformations and final product analysis using Reflexion and Debate
    
    # Sub-task 4: Identify product 4 formed by diazotization of product 3
    cot_instruction_4 = (
        "Sub-task 4: Identify the structure of product 4 formed by treating product 3 with NaNO2 and HBF4, "
        "involving diazotization of the amino group and formation of the diazonium salt."
    )
    cot_agent_4 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    thinking4, answer4 = await cot_agent_4([taskInfo, thinking3, answer3], cot_instruction_4, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_4.id}, diazotization step, thinking: {thinking4.content}; answer: {answer4.content}")
    sub_tasks.append(f"Sub-task 4 output: thinking - {thinking4.content}; answer - {answer4.content}")
    print("Step 4: ", sub_tasks[-1])
    
    # Sub-task 5: Determine final product 5 formed by heating product 4 and treating with anisole (Sandmeyer-type reaction)
    debate_instruction_5 = (
        "Sub-task 5: Based on product 4, determine the final product 5 formed by heating product 4 and then treating it with anisole, "
        "involving the Sandmeyer-type reaction or related coupling to form a biphenyl derivative with anisole substituent."
    )
    debate_roles = ["Proposer", "Opponent"]
    debate_agents_5 = [LLMAgentBase(["thinking", "answer"], "Debate Agent", model=self.node_model, role=role, temperature=0.5) for role in debate_roles]
    N_max_5 = self.max_round
    all_thinking5 = [[] for _ in range(N_max_5)]
    all_answer5 = [[] for _ in range(N_max_5)]
    for r in range(N_max_5):
        for i, agent in enumerate(debate_agents_5):
            if r == 0:
                thinking5, answer5 = await agent([taskInfo, thinking4, answer4], debate_instruction_5, r, is_sub_task=True)
            else:
                input_infos_5 = [taskInfo, thinking4, answer4] + all_thinking5[r-1] + all_answer5[r-1]
                thinking5, answer5 = await agent(input_infos_5, debate_instruction_5, r, is_sub_task=True)
            agents.append(f"Debate agent {agent.id}, round {r}, proposing final product structure, thinking: {thinking5.content}; answer: {answer5.content}")
            all_thinking5[r].append(thinking5)
            all_answer5[r].append(answer5)
    
    # Sub-task 6: Analyze substitution pattern and regiochemistry to match final product with given choices using Reflexion
    cot_reflect_instruction_6 = (
        "Sub-task 6: Analyze the substitution pattern and regiochemistry of the final product 5 to match it with one of the given choices: "
        "3'-bromo-2-methoxy-1,1'-biphenyl, 3-bromo-4'-methoxy-1,1'-biphenyl, 4-bromo-4'-methoxy-1,1'-biphenyl, or 3-bromo-4'-fluoro-1,1'-biphenyl, "
        "using all previous structural information and reaction mechanisms."
    )
    final_decision_agent_6 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking6, answer6 = await final_decision_agent_6(
        [taskInfo] + all_thinking5[-1] + all_answer5[-1] + [thinking1, answer1, thinking2_final, answermapping_2[answer2_final], thinking3, answer3, thinking4, answer4],
        cot_reflect_instruction_6, is_sub_task=True
    )
    agents.append(f"Final Decision agent, analyzing final product regiochemistry, thinking: {thinking6.content}; answer: {answer6.content}")
    sub_tasks.append(f"Sub-task 6 output: thinking - {thinking6.content}; answer - {answer6.content}")
    print("Step 6: ", sub_tasks[-1])
    
    final_answer = await self.make_final_answer(thinking6, answer6, sub_tasks, agents)
    return final_answer
