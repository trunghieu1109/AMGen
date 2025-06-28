async def forward_22(self, taskInfo):
    from collections import Counter
    print("Task Requirement: ", taskInfo)
    sub_tasks = []
    agents = []

    # Stage 1: Analyze and Interpret Inputs
    # Sub-task 1: Analyze structure and reactive sites of ((2,2-dimethylbut-3-en-1-yl)oxy)benzene
    cot_instruction_1 = (
        "Sub-task 1: Analyze the structure and functional groups of the starting material ((2,2-dimethylbut-3-en-1-yl)oxy)benzene, "
        "identifying reactive sites relevant to reaction with hydrogen bromide (HBr). This includes understanding the alkene moiety, ether linkage, and aromatic ring context."
    )
    cot_agent_1 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    thinking1, answer1 = await cot_agent_1([taskInfo], cot_instruction_1, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_1.id}, analyzing starting material structure, thinking: {thinking1.content}; answer: {answer1.content}")
    sub_tasks.append(f"Sub-task 1 output: thinking - {thinking1.content}; answer - {answer1.content}")
    print("Step 1: ", sub_tasks[-1])

    # Sub-task 2: Analyze chemical properties and reaction behavior of HBr with alkenes and ethers
    cot_instruction_2 = (
        "Sub-task 2: Analyze the chemical properties and typical reaction behavior of hydrogen bromide (HBr) with alkenes and ethers, "
        "focusing on possible addition, substitution, or rearrangement reactions under the experimental conditions."
    )
    cot_agent_2 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    thinking2, answer2 = await cot_agent_2([taskInfo], cot_instruction_2, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_2.id}, analyzing HBr reaction behavior, thinking: {thinking2.content}; answer: {answer2.content}")
    sub_tasks.append(f"Sub-task 2 output: thinking - {thinking2.content}; answer - {answer2.content}")
    print("Step 2: ", sub_tasks[-1])

    # Sub-task 3: Interpret TLC observation of diminished reactant spot and two new spots
    cot_instruction_3 = (
        "Sub-task 3: Interpret the TLC observation that the reactant spot diminished and two new spots appeared, "
        "inferring that two distinct products formed from the reaction of the starting material with HBr."
    )
    cot_agent_3 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    thinking3, answer3 = await cot_agent_3([taskInfo, thinking1, answer1, thinking2, answer2], cot_instruction_3, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_3.id}, interpreting TLC results, thinking: {thinking3.content}; answer: {answer3.content}")
    sub_tasks.append(f"Sub-task 3 output: thinking - {thinking3.content}; answer - {answer3.content}")
    print("Step 3: ", sub_tasks[-1])

    # Stage 2: Predict and Evaluate Products
    # Sub-task 4: Predict plausible product structures formed from reaction
    cot_instruction_4 = (
        "Sub-task 4: Predict plausible product structures formed from the reaction of ((2,2-dimethylbut-3-en-1-yl)oxy)benzene with HBr, "
        "considering electrophilic addition to the alkene, possible intramolecular cyclization, and substitution reactions, consistent with the formation of two products."
    )
    cot_agent_4 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    thinking4, answer4 = await cot_agent_4([taskInfo, thinking1, answer1, thinking2, answer2, thinking3, answer3], cot_instruction_4, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_4.id}, predicting plausible products, thinking: {thinking4.content}; answer: {answer4.content}")
    sub_tasks.append(f"Sub-task 4 output: thinking - {thinking4.content}; answer - {answer4.content}")
    print("Step 4: ", sub_tasks[-1])

    # Sub-task 5: Evaluate each given product choice against predicted products using Debate
    debate_instruction_5 = (
        "Sub-task 5: Evaluate each of the given product choices (choice1 to choice4) against the predicted plausible products from Sub-task 4, "
        "assessing structural consistency with expected reaction mechanisms and TLC results."
    )
    debate_agents_5 = [
        LLMAgentBase(["thinking", "answer"], "Debate Agent", model=self.node_model, role=role, temperature=0.5)
        for role in self.debate_role
    ]
    N_max_5 = self.max_round
    all_thinking5 = [[] for _ in range(N_max_5)]
    all_answer5 = [[] for _ in range(N_max_5)]

    for r in range(N_max_5):
        for i, agent in enumerate(debate_agents_5):
            if r == 0:
                thinking5, answer5 = await agent(
                    [taskInfo, thinking4, answer4], debate_instruction_5, r, is_sub_task=True
                )
            else:
                input_infos_5 = [taskInfo, thinking4, answer4] + all_thinking5[r-1] + all_answer5[r-1]
                thinking5, answer5 = await agent(input_infos_5, debate_instruction_5, r, is_sub_task=True)
            agents.append(f"Debate agent {agent.id}, round {r}, evaluating product choices, thinking: {thinking5.content}; answer: {answer5.content}")
            all_thinking5[r].append(thinking5)
            all_answer5[r].append(answer5)

    final_decision_agent_5 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking5, answer5 = await final_decision_agent_5(
        [taskInfo] + all_thinking5[-1] + all_answer5[-1],
        "Sub-task 5: Make final decision on which product choices best correspond to the predicted products.",
        is_sub_task=True
    )
    agents.append(f"Final Decision agent, deciding best product choices, thinking: {thinking5.content}; answer: {answer5.content}")
    sub_tasks.append(f"Sub-task 5 output: thinking - {thinking5.content}; answer - {answer5.content}")
    print("Step 5: ", sub_tasks[-1])

    # Sub-task 6: Determine best choice(s) with reasoned justification
    cot_reflect_instruction_6 = (
        "Sub-task 6: Based on evaluation in Sub-task 5, determine which choice(s) among the given options best correspond to the predicted products, "
        "providing a reasoned justification based on chemical logic and reaction pathways."
    )
    cot_agent_6 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    thinking6, answer6 = await cot_agent_6([taskInfo, thinking5, answer5], cot_reflect_instruction_6, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_6.id}, determining best product choices with justification, thinking: {thinking6.content}; answer: {answer6.content}")
    sub_tasks.append(f"Sub-task 6 output: thinking - {thinking6.content}; answer - {answer6.content}")
    print("Step 6: ", sub_tasks[-1])

    final_answer = await self.make_final_answer(thinking6, answer6, sub_tasks, agents)
    return final_answer
