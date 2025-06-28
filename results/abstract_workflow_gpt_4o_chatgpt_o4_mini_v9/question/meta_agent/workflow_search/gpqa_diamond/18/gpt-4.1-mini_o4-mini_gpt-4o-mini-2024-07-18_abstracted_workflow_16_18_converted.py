async def forward_18(self, taskInfo):
    from collections import Counter
    print("Task Requirement: ", taskInfo)
    sub_tasks = []
    agents = []

    # Stage 1: Analyze Michael reaction mechanism and reactants characterization

    # Sub-task 1: Analyze the Michael reaction mechanism focusing on nucleophile addition to alpha,beta-unsaturated carbonyl compounds
    cot_instruction_1 = (
        "Sub-task 1: Analyze the Michael reaction mechanism focusing on nucleophile addition to alpha,beta-unsaturated carbonyl compounds, "
        "to understand the general reaction pathway and the nature of the new carbon-carbon bond formed."
    )
    cot_agent_1 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    thinking1, answer1 = await cot_agent_1([taskInfo], cot_instruction_1, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_1.id}, analyzing Michael reaction mechanism, thinking: {thinking1.content}; answer: {answer1.content}")
    sub_tasks.append(f"Sub-task 1 output: thinking - {thinking1.content}; answer - {answer1.content}")
    print("Step 1: ", sub_tasks[-1])

    # Sub-task 2: Identify and characterize reactants and reagents in reaction A
    cot_instruction_2 = (
        "Sub-task 2: Identify and characterize the reactants and reagents in reaction A (methyl 2-oxocyclohexane-1-carboxylate, NaOEt, THF, 2,4-dimethyl-1-(vinylsulfinyl)benzene), "
        "including their functional groups and reactive sites relevant to the Michael reaction mechanism established in Sub-task 1."
    )
    cot_agent_2 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    thinking2, answer2 = await cot_agent_2([taskInfo, thinking1, answer1], cot_instruction_2, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_2.id}, characterizing reactants/reagents in reaction A, thinking: {thinking2.content}; answer: {answer2.content}")
    sub_tasks.append(f"Sub-task 2 output: thinking - {thinking2.content}; answer - {answer2.content}")
    print("Step 2: ", sub_tasks[-1])

    # Sub-task 3: Identify and characterize reactants and reagents in reaction B
    cot_instruction_3 = (
        "Sub-task 3: Identify and characterize the reactants and reagents in reaction B (ethyl 2-ethylbutanoate, NaH, THF, methyl 2-cyclopentylidene-2-phenylacetate), "
        "including their functional groups and reactive sites relevant to the Michael reaction mechanism established in Sub-task 1."
    )
    cot_agent_3 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    thinking3, answer3 = await cot_agent_3([taskInfo, thinking1, answer1], cot_instruction_3, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_3.id}, characterizing reactants/reagents in reaction B, thinking: {thinking3.content}; answer: {answer3.content}")
    sub_tasks.append(f"Sub-task 3 output: thinking - {thinking3.content}; answer - {answer3.content}")
    print("Step 3: ", sub_tasks[-1])

    # Sub-task 4: Determine expected Michael addition product structure for reaction A
    cot_instruction_4 = (
        "Sub-task 4: Determine the expected Michael addition product structure for reaction A by applying the Michael reaction mechanism "
        "to the characterized reactants and reagents from Sub-task 2, focusing on the site of nucleophilic attack and the resulting carbon-carbon bond formation."
    )
    cot_agent_4 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    thinking4, answer4 = await cot_agent_4([taskInfo, thinking2, answer2], cot_instruction_4, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_4.id}, determining product structure for reaction A, thinking: {thinking4.content}; answer: {answer4.content}")
    sub_tasks.append(f"Sub-task 4 output: thinking - {thinking4.content}; answer - {answer4.content}")
    print("Step 4: ", sub_tasks[-1])

    # Sub-task 5: Determine expected Michael addition product structure for reaction B
    cot_instruction_5 = (
        "Sub-task 5: Determine the expected Michael addition product structure for reaction B by applying the Michael reaction mechanism "
        "to the characterized reactants and reagents from Sub-task 3, focusing on the site of nucleophilic attack and the resulting carbon-carbon bond formation."
    )
    cot_agent_5 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    thinking5, answer5 = await cot_agent_5([taskInfo, thinking3, answer3], cot_instruction_5, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_5.id}, determining product structure for reaction B, thinking: {thinking5.content}; answer: {answer5.content}")
    sub_tasks.append(f"Sub-task 5 output: thinking - {thinking5.content}; answer - {answer5.content}")
    print("Step 5: ", sub_tasks[-1])

    # Stage 2: Compare predicted products with candidate choices and select correct overall choice

    # Sub-task 6: Compare predicted product for reaction A with candidate choices
    cot_instruction_6 = (
        "Sub-task 6: Compare the predicted product structure for reaction A (from Sub-task 4) with the candidate product structures given in choices 1 to 4, "
        "to identify which choice correctly corresponds to product A."
    )
    cot_agent_6 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    thinking6, answer6 = await cot_agent_6([taskInfo, thinking4, answer4], cot_instruction_6, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_6.id}, comparing predicted product A with choices, thinking: {thinking6.content}; answer: {answer6.content}")
    sub_tasks.append(f"Sub-task 6 output: thinking - {thinking6.content}; answer - {answer6.content}")
    print("Step 6: ", sub_tasks[-1])

    # Sub-task 7: Compare predicted product for reaction B with candidate choices
    cot_instruction_7 = (
        "Sub-task 7: Compare the predicted product structure for reaction B (from Sub-task 5) with the candidate product structures given in choices 1 to 4, "
        "to identify which choice correctly corresponds to product B."
    )
    cot_agent_7 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    thinking7, answer7 = await cot_agent_7([taskInfo, thinking5, answer5], cot_instruction_7, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_7.id}, comparing predicted product B with choices, thinking: {thinking7.content}; answer: {answer7.content}")
    sub_tasks.append(f"Sub-task 7 output: thinking - {thinking7.content}; answer - {answer7.content}")
    print("Step 7: ", sub_tasks[-1])

    # Sub-task 8: Integrate results from subtask 6 and 7 to select correct overall choice
    debate_instruction_8 = (
        "Sub-task 8: Integrate the results from Sub-task 6 and Sub-task 7 to select the correct overall choice (1 to 4) that correctly identifies both products A and B, "
        "ensuring consistency with the Michael reaction mechanism and the given reactants and reagents."
    )
    debate_agents_8 = [LLMAgentBase(["thinking", "answer"], "Debate Agent", model=self.node_model, role=role, temperature=0.5) for role in self.debate_role]
    N_max_8 = self.max_round
    all_thinking8 = [[] for _ in range(N_max_8)]
    all_answer8 = [[] for _ in range(N_max_8)]

    for r in range(N_max_8):
        for i, agent in enumerate(debate_agents_8):
            if r == 0:
                thinking8, answer8 = await agent([taskInfo, thinking6, answer6, thinking7, answer7], debate_instruction_8, r, is_sub_task=True)
            else:
                input_infos_8 = [taskInfo, thinking6, answer6, thinking7, answer7] + all_thinking8[r-1] + all_answer8[r-1]
                thinking8, answer8 = await agent(input_infos_8, debate_instruction_8, r, is_sub_task=True)
            agents.append(f"Debate agent {agent.id}, round {r}, integrating results to select correct overall choice, thinking: {thinking8.content}; answer: {answer8.content}")
            all_thinking8[r].append(thinking8)
            all_answer8[r].append(answer8)

    final_decision_agent_8 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking8, answer8 = await final_decision_agent_8([taskInfo] + all_thinking8[-1] + all_answer8[-1], "Sub-task 8: Make final decision on the correct overall choice identifying both products A and B.", is_sub_task=True)
    agents.append(f"Final Decision agent, making final choice decision, thinking: {thinking8.content}; answer: {answer8.content}")
    sub_tasks.append(f"Sub-task 8 output: thinking - {thinking8.content}; answer - {answer8.content}")
    print("Step 8: ", sub_tasks[-1])

    final_answer = await self.make_final_answer(thinking8, answer8, sub_tasks, agents)
    return final_answer
