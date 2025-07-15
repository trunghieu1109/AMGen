async def forward_185(self, taskInfo):
    from collections import Counter
    sub_tasks = []
    agents = []
    logs = []

    # Sub-task 1: SC-CoT to extract and characterize the starting substrate
    sc_instruction1 = (
        "Sub-task 1: Extract and characterize the starting substrate (1S,4R)-2-vinyl-2-azabicyclo[2.2.1]hept-5-ene by identifying the azabicyclo[2.2.1] framework, "
        "the bridgehead nitrogen at C-2, the vinyl substituent, and the 1S,4R stereochemistry."
    )
    N1 = self.max_sc
    sc_agents1 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent",
                                model=self.node_model, temperature=0.5)
                  for _ in range(N1)]
    possible_think1 = []
    possible_ans1 = []
    subtask1_desc = {
        "subtask_id": "subtask_1",
        "instruction": sc_instruction1,
        "context": ["user query"],
        "agent_collaboration": "SC_CoT"
    }
    for i in range(N1):
        thinking1_i, answer1_i = await sc_agents1[i]([taskInfo], sc_instruction1, is_sub_task=True)
        agents.append(
            f"CoT-SC agent {sc_agents1[i].id}, extracting substrate characteristics, thinking: {thinking1_i.content}; answer: {answer1_i.content}"
        )
        possible_think1.append(thinking1_i)
        possible_ans1.append(answer1_i)
    final_decision1 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent",
                                    model=self.node_model, temperature=0.0)
    final_instr1 = (
        "Sub-task 1: Synthesize and choose the most consistent and correct characterization of the starting substrate. "
        "Given all the above thinking and answers, find the most consistent and correct solution for Sub-task 1."
    )
    thinking1_fd, answer1_fd = await final_decision1(
        [taskInfo] + possible_think1 + possible_ans1,
        final_instr1,
        is_sub_task=True
    )
    agents.append(
        f"Final Decision Agent {final_decision1.id}, synthesizing substrate characterization, thinking: {thinking1_fd.content}; answer: {answer1_fd.content}"
    )
    sub_tasks.append(f"Sub-task 1 output: thinking - {thinking1_fd.content}; answer - {answer1_fd.content}")
    subtask1_desc['response'] = {"thinking": thinking1_fd, "answer": answer1_fd}
    logs.append(subtask1_desc)
    print("Step 1: ", sub_tasks[-1])

    # Sub-task 2: SC-CoT to extract and characterize the reaction and product options
    sc_instruction2 = (
        "Sub-task 2: Extract and characterize the Cope rearrangement mechanism and list the four cyclopenta[c]pyridine isomer names as products."
    )
    N2 = self.max_sc
    sc_agents2 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent",
                                model=self.node_model, temperature=0.5)
                  for _ in range(N2)]
    possible_think2 = []
    possible_ans2 = []
    subtask2_desc = {
        "subtask_id": "subtask_2",
        "instruction": sc_instruction2,
        "context": ["user query"],
        "agent_collaboration": "SC_CoT"
    }
    for i in range(N2):
        thinking2_i, answer2_i = await sc_agents2[i]([taskInfo], sc_instruction2, is_sub_task=True)
        agents.append(
            f"CoT-SC agent {sc_agents2[i].id}, summarizing mechanism and options, thinking: {thinking2_i.content}; answer: {answer2_i.content}"
        )
        possible_think2.append(thinking2_i)
        possible_ans2.append(answer2_i)
    final_decision2 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent",
                                    model=self.node_model, temperature=0.0)
    final_instr2 = (
        "Sub-task 2: Synthesize and choose the most consistent and correct summary of the Cope mechanism and product names. "
        "Given all the above thinking and answers, find the most consistent and correct solution for Sub-task 2."
    )
    thinking2_fd, answer2_fd = await final_decision2(
        [taskInfo] + possible_think2 + possible_ans2,
        final_instr2,
        is_sub_task=True
    )
    agents.append(
        f"Final Decision Agent {final_decision2.id}, synthesizing mechanism summary, thinking: {thinking2_fd.content}; answer: {answer2_fd.content}"
    )
    sub_tasks.append(f"Sub-task 2 output: thinking - {thinking2_fd.content}; answer - {answer2_fd.content}")
    subtask2_desc['response'] = {"thinking": thinking2_fd, "answer": answer2_fd}
    logs.append(subtask2_desc)
    print("Step 2: ", sub_tasks[-1])

    # Sub-task 3: SC-CoT to apply Cope rearrangement and deduce product skeleton
    sc_instruction3 = (
        "Sub-task 3: Apply the Cope rearrangement to the characterized substrate and deduce the new ring connectivity, position of the double bond, and stereochemical outcome."
    )
    N3 = self.max_sc
    sc_agents3 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent",
                                model=self.node_model, temperature=0.5)
                  for _ in range(N3)]
    possible_think3 = []
    possible_ans3 = []
    subtask3_desc = {
        "subtask_id": "subtask_3",
        "instruction": sc_instruction3,
        "context": ["user query", thinking1_fd, answer1_fd, thinking2_fd, answer2_fd],
        "agent_collaboration": "SC_CoT"
    }
    for i in range(N3):
        thinking3_i, answer3_i = await sc_agents3[i](
            [taskInfo, thinking1_fd, answer1_fd, thinking2_fd, answer2_fd],
            sc_instruction3,
            is_sub_task=True
        )
        agents.append(
            f"CoT-SC agent {sc_agents3[i].id}, deducing product skeleton, thinking: {thinking3_i.content}; answer: {answer3_i.content}"
        )
        possible_think3.append(thinking3_i)
        possible_ans3.append(answer3_i)
    final_decision3 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent",
                                    model=self.node_model, temperature=0.0)
    final_instr3 = (
        "Sub-task 3: Synthesize and choose the most consistent and correct deduction of the product skeleton. "
        "Given all the above thinking and answers, find the most consistent and correct solution for Sub-task 3."
    )
    thinking3_fd, answer3_fd = await final_decision3(
        [taskInfo, thinking1_fd, answer1_fd, thinking2_fd, answer2_fd] + possible_think3 + possible_ans3,
        final_instr3,
        is_sub_task=True
    )
    agents.append(
        f"Final Decision Agent {final_decision3.id}, synthesizing product deduction, thinking: {thinking3_fd.content}; answer: {answer3_fd.content}"
    )
    sub_tasks.append(f"Sub-task 3 output: thinking - {thinking3_fd.content}; answer - {answer3_fd.content}")
    subtask3_desc['response'] = {"thinking": thinking3_fd, "answer": answer3_fd}
    logs.append(subtask3_desc)
    print("Step 3: ", sub_tasks[-1])

    # Sub-task 4: Reflexion to map product skeleton onto IUPAC names
    reflect_inst = (
        "Given previous attempts and feedback, carefully consider where you could go wrong in your latest attempt. "
        "Using insights from previous attempts, try to solve the task better."
    )
    cot_reflect_instruction = (
        "Sub-task 4: Map the deduced product skeleton and stereochemistry onto the four given IUPAC names, determining which name matches. "
        + reflect_inst
    )
    cot_agent4 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent",
                              model=self.node_model, temperature=0.0)
    critic_agent4 = LLMAgentBase(["feedback", "correct"], "Critic Agent",
                                  model=self.node_model, temperature=0.0)
    inputs4 = [taskInfo, thinking3_fd, answer3_fd]
    subtask4_desc = {
        "subtask_id": "subtask_4",
        "instruction": cot_reflect_instruction,
        "context": ["user query", thinking3_fd, answer3_fd],
        "agent_collaboration": "Reflexion"
    }
    thinking4, answer4 = await cot_agent4(inputs4, cot_reflect_instruction, 0, is_sub_task=True)
    agents.append(
        f"Reflexion CoT agent {cot_agent4.id}, mapping product to IUPAC names, thinking: {thinking4.content}; answer: {answer4.content}"
    )
    critic_inst4 = (
        "Please review the answer above and criticize on where might be wrong. "
        "If you are absolutely sure it is correct, output exactly 'True' in 'correct'"
    )
    for i in range(self.max_round):
        feedback4, correct4 = await critic_agent4(
            [taskInfo, thinking4, answer4],
            "Please review and provide the limitations of provided solution. " + critic_inst4,
            i,
            is_sub_task=True
        )
        agents.append(
            f"Critic agent {critic_agent4.id}, providing feedback, thinking: {feedback4.content}; answer: {correct4.content}"
        )
        if correct4.content == "True":
            break
        inputs4.extend([thinking4, answer4, feedback4])
        thinking4, answer4 = await cot_agent4(inputs4, cot_reflect_instruction, i + 1, is_sub_task=True)
        agents.append(
            f"Reflexion CoT agent {cot_agent4.id}, refining mapping to names, thinking: {thinking4.content}; answer: {answer4.content}"
        )
    sub_tasks.append(f"Sub-task 4 output: thinking - {thinking4.content}; answer - {answer4.content}")
    subtask4_desc['response'] = {"thinking": thinking4, "answer": answer4}
    logs.append(subtask4_desc)
    print("Step 4: ", sub_tasks[-1])

    # Sub-task 5: Debate to evaluate and justify the single correct product choice
    debate_instr = (
        "Given solutions to the problem from other agents, consider their opinions as additional advice. "
        "Please think carefully and provide an updated answer."
    )
    debate_instruction5 = (
        "Sub-task 5: Evaluate and justify the single correct product choice by comparing unsaturation levels, tautomeric form, and numbering to eliminate the others. "
        + debate_instr
    )
    debate_agents5 = [LLMAgentBase(["thinking", "answer"], "Debate Agent",
                                     model=self.node_model, role=role, temperature=0.5)
                       for role in self.debate_role]
    all_thinking5 = [[] for _ in range(self.max_round)]
    all_answer5 = [[] for _ in range(self.max_round)]
    subtask5_desc = {
        "subtask_id": "subtask_5",
        "instruction": debate_instruction5,
        "context": ["user query", thinking4, answer4],
        "agent_collaboration": "Debate"
    }
    for r in range(self.max_round):
        for i, agent in enumerate(debate_agents5):
            if r == 0:
                thinking5_i, answer5_i = await agent(
                    [taskInfo, thinking4, answer4],
                    debate_instruction5,
                    r,
                    is_sub_task=True
                )
            else:
                inputs5 = [taskInfo, thinking4, answer4] + all_thinking5[r-1] + all_answer5[r-1]
                thinking5_i, answer5_i = await agent(
                    inputs5,
                    debate_instruction5,
                    r,
                    is_sub_task=True
                )
            agents.append(
                f"Debate agent {agent.id}, round {r}, thinking: {thinking5_i.content}; answer: {answer5_i.content}"
            )
            all_thinking5[r].append(thinking5_i)
            all_answer5[r].append(answer5_i)
    final_decision5 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent",
                                    model=self.node_model, temperature=0.0)
    final_instr5 = (
        "Sub-task 5: Given all the above thinking and answers, reason over them carefully and provide a final answer."
    )
    thinking5_fd, answer5_fd = await final_decision5(
        [taskInfo, thinking4, answer4] + all_thinking5[-1] + all_answer5[-1],
        final_instr5,
        is_sub_task=True
    )
    agents.append(
        f"Final Decision agent {final_decision5.id}, final justification, thinking: {thinking5_fd.content}; answer: {answer5_fd.content}"
    )
    sub_tasks.append(f"Sub-task 5 output: thinking - {thinking5_fd.content}; answer - {answer5_fd.content}")
    subtask5_desc['response'] = {"thinking": thinking5_fd, "answer": answer5_fd}
    logs.append(subtask5_desc)
    print("Step 5: ", sub_tasks[-1])

    final_answer = await self.make_final_answer(thinking5_fd, answer5_fd, sub_tasks, agents)
    return final_answer, logs