async def forward_99(self, taskInfo):
    from collections import Counter
    
    print("Task Requirement: ", taskInfo)
    
    sub_tasks = []
    agents = []

    # Stage 1: Identify and determine structures of compounds A to H sequentially using Chain-of-Thought for each subtask

    # Sub-task 1: Identify Compound A (C3H6) structure and isomers
    cot_instruction_1 = "Sub-task 1: Identify and determine the structure and nature of Compound A (C3H6), including possible isomers, to understand the starting material for the reaction sequence." 
    cot_agent_1 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    thinking1, answer1 = await cot_agent_1([taskInfo], cot_instruction_1, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_1.id}, identifying Compound A, thinking: {thinking1.content}; answer: {answer1.content}")
    sub_tasks.append(f"Sub-task 1 output: thinking - {thinking1.content}; answer - {answer1.content}")
    print("Step 1: ", sub_tasks[-1])

    # Sub-task 2: Analyze bromination of Compound A to form Compound B
    cot_instruction_2 = "Sub-task 2: Analyze the bromination of Compound A in the presence of carbon tetrachloride to determine the structure and properties of Compound B." 
    cot_agent_2 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    thinking2, answer2 = await cot_agent_2([taskInfo, thinking1, answer1], cot_instruction_2, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_2.id}, analyzing bromination to form Compound B, thinking: {thinking2.content}; answer: {answer2.content}")
    sub_tasks.append(f"Sub-task 2 output: thinking - {thinking2.content}; answer - {answer2.content}")
    print("Step 2: ", sub_tasks[-1])

    # Sub-task 3: Examine reaction of Compound B with alcoholic KOH to identify Compound C
    cot_instruction_3 = "Sub-task 3: Examine the reaction of Compound B with alcoholic KOH to identify Compound C, including its physical state and flammability." 
    cot_agent_3 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    thinking3, answer3 = await cot_agent_3([taskInfo, thinking2, answer2], cot_instruction_3, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_3.id}, examining reaction to form Compound C, thinking: {thinking3.content}; answer: {answer3.content}")
    sub_tasks.append(f"Sub-task 3 output: thinking - {thinking3.content}; answer - {answer3.content}")
    print("Step 3: ", sub_tasks[-1])

    # Sub-task 4: Determine Compound D formed by passing Compound C through red-hot iron tube
    cot_instruction_4 = "Sub-task 4: Determine the product Compound D formed by passing Compound C through a red-hot iron tube, including its structural features relevant to NMR spectra." 
    cot_agent_4 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    thinking4, answer4 = await cot_agent_4([taskInfo, thinking3, answer3], cot_instruction_4, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_4.id}, determining Compound D and NMR features, thinking: {thinking4.content}; answer: {answer4.content}")
    sub_tasks.append(f"Sub-task 4 output: thinking - {thinking4.content}; answer - {answer4.content}")
    print("Step 4: ", sub_tasks[-1])

    # Sub-task 5: Analyze reaction of Compound D with two strong acids to form Compound E
    cot_instruction_5 = "Sub-task 5: Analyze the reaction of Compound D with a mixture of two strong acids to form Compound E, identifying the functional groups and chemical nature of E." 
    cot_agent_5 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    thinking5, answer5 = await cot_agent_5([taskInfo, thinking4, answer4], cot_instruction_5, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_5.id}, analyzing formation of Compound E, thinking: {thinking5.content}; answer: {answer5.content}")
    sub_tasks.append(f"Sub-task 5 output: thinking - {thinking5.content}; answer - {answer5.content}")
    print("Step 5: ", sub_tasks[-1])

    # Sub-task 6: Identify Compound F formed when Compound E reacts with iron scrap and HCl
    cot_instruction_6 = "Sub-task 6: Identify Compound F formed when Compound E reacts with iron scrap and hydrochloric acid, including its common uses such as in dye synthesis." 
    cot_agent_6 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    thinking6, answer6 = await cot_agent_6([taskInfo, thinking5, answer5], cot_instruction_6, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_6.id}, identifying Compound F and uses, thinking: {thinking6.content}; answer: {answer6.content}")
    sub_tasks.append(f"Sub-task 6 output: thinking - {thinking6.content}; answer - {answer6.content}")
    print("Step 6: ", sub_tasks[-1])

    # Sub-task 7: Determine structure and properties of Compound G formed by reaction of F with nitrous acid
    cot_instruction_7 = "Sub-task 7: Determine the structure and properties of Compound G formed by the reaction of Compound F with nitrous acid." 
    cot_agent_7 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    thinking7, answer7 = await cot_agent_7([taskInfo, thinking6, answer6], cot_instruction_7, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_7.id}, determining Compound G, thinking: {thinking7.content}; answer: {answer7.content}")
    sub_tasks.append(f"Sub-task 7 output: thinking - {thinking7.content}; answer - {answer7.content}")
    print("Step 7: ", sub_tasks[-1])

    # Sub-task 8: Analyze reaction of Compound G with NaOH to form Compound H
    cot_instruction_8 = "Sub-task 8: Analyze the reaction of Compound G with sodium hydroxide to form Compound H, including its chemical properties and reactions such as color change with ferric chloride." 
    cot_agent_8 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    thinking8, answer8 = await cot_agent_8([taskInfo, thinking7, answer7], cot_instruction_8, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_8.id}, analyzing Compound H and its properties, thinking: {thinking8.content}; answer: {answer8.content}")
    sub_tasks.append(f"Sub-task 8 output: thinking - {thinking8.content}; answer - {answer8.content}")
    print("Step 8: ", sub_tasks[-1])

    # Stage 2: Evaluate the correctness of the statements about compounds D, F, H, and C

    # Sub-task 9: Evaluate statement about Compound D NMR spectra (two singlets)
    cot_instruction_9 = "Sub-task 9: Evaluate the validity of the statement that Compound D gives two singlets in the 1H NMR spectra based on the structure and symmetry of D." 
    cot_agent_9 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    thinking9, answer9 = await cot_agent_9([taskInfo, thinking4, answer4], cot_instruction_9, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_9.id}, evaluating NMR statement for Compound D, thinking: {thinking9.content}; answer: {answer9.content}")
    sub_tasks.append(f"Sub-task 9 output: thinking - {thinking9.content}; answer - {answer9.content}")
    print("Step 9: ", sub_tasks[-1])

    # Sub-task 10: Assess statement that Compound F is used for dye synthesis
    cot_instruction_10 = "Sub-task 10: Assess the correctness of the statement that Compound F is used for the synthesis of dyes, based on the identified nature and common applications of F." 
    cot_agent_10 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    thinking10, answer10 = await cot_agent_10([taskInfo, thinking6, answer6], cot_instruction_10, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_10.id}, assessing dye synthesis use of Compound F, thinking: {thinking10.content}; answer: {answer10.content}")
    sub_tasks.append(f"Sub-task 10 output: thinking - {thinking10.content}; answer - {answer10.content}")
    print("Step 10: ", sub_tasks[-1])

    # Sub-task 11: Verify statement that Compound H gives yellow color with ferric chloride
    cot_instruction_11 = "Sub-task 11: Verify the statement that Compound H gives a yellow color with the addition of ferric chloride solution, based on the chemical properties of H." 
    cot_agent_11 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    thinking11, answer11 = await cot_agent_11([taskInfo, thinking8, answer8], cot_instruction_11, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_11.id}, verifying color reaction of Compound H, thinking: {thinking11.content}; answer: {answer11.content}")
    sub_tasks.append(f"Sub-task 11 output: thinking - {thinking11.content}; answer - {answer11.content}")
    print("Step 11: ", sub_tasks[-1])

    # Sub-task 12: Check statement that Compound C is a flammable gas
    cot_instruction_12 = "Sub-task 12: Check the statement that Compound C is a flammable gas, based on the physical state and properties determined for C." 
    cot_agent_12 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    thinking12, answer12 = await cot_agent_12([taskInfo, thinking3, answer3], cot_instruction_12, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_12.id}, checking flammability and state of Compound C, thinking: {thinking12.content}; answer: {answer12.content}")
    sub_tasks.append(f"Sub-task 12 output: thinking - {thinking12.content}; answer - {answer12.content}")
    print("Step 12: ", sub_tasks[-1])

    # Sub-task 13: Integrate evaluations to identify incorrect statement using Debate pattern
    debate_instruction_13 = "Sub-task 13: Integrate the evaluations of all statements (Sub-tasks 9 to 12) to identify which statement about the products in the reaction sequence is incorrect." 
    debate_roles = ["Proponent", "Opponent"]
    debate_agents_13 = [LLMAgentBase(["thinking", "answer"], "Debate Agent", model=self.node_model, role=role, temperature=0.5) for role in debate_roles]
    N_max_13 = self.max_round
    all_thinking13 = [[] for _ in range(N_max_13)]
    all_answer13 = [[] for _ in range(N_max_13)]

    for r in range(N_max_13):
        for i, agent in enumerate(debate_agents_13):
            if r == 0:
                thinking13, answer13 = await agent(
                    [taskInfo, thinking9, answer9, thinking10, answer10, thinking11, answer11, thinking12, answer12],
                    debate_instruction_13, r, is_sub_task=True)
            else:
                input_infos_13 = [taskInfo, thinking9, answer9, thinking10, answer10, thinking11, answer11, thinking12, answer12] + all_thinking13[r-1] + all_answer13[r-1]
                thinking13, answer13 = await agent(input_infos_13, debate_instruction_13, r, is_sub_task=True)
            agents.append(f"Debate agent {agent.id}, round {r}, debating incorrect statement, thinking: {thinking13.content}; answer: {answer13.content}")
            all_thinking13[r].append(thinking13)
            all_answer13[r].append(answer13)

    final_decision_agent_13 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking13, answer13 = await final_decision_agent_13(
        [taskInfo] + all_thinking13[-1] + all_answer13[-1],
        "Sub-task 13: Make final decision on which statement is incorrect.",
        is_sub_task=True)
    agents.append(f"Final Decision agent, deciding incorrect statement, thinking: {thinking13.content}; answer: {answer13.content}")
    sub_tasks.append(f"Sub-task 13 output: thinking - {thinking13.content}; answer - {answer13.content}")
    print("Step 13: ", sub_tasks[-1])

    final_answer = await self.make_final_answer(thinking13, answer13, sub_tasks, agents)
    return final_answer
