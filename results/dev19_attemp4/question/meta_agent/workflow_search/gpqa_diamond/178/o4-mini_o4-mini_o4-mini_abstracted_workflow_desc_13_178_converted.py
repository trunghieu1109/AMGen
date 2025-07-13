async def forward_178(self, taskInfo):
    from collections import Counter
    print("Task Requirement:", taskInfo)
    sub_tasks = []
    agents = []
    logs = []

    # Stage 1: Formalize criteria (SC_CoT)
    cot_sc_instruction = (
        "Sub-task 1: Define quantum‐mechanical criteria: unitarity (U†U=I), Hermiticity (A†=A), "
        "anti-Hermiticity (X† = -X => e^X unitary), density matrix (eigenvalues ≥0, trace=1)."
    )
    N = self.max_sc
    cot_agents = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N)]
    possible_thinkings = []
    possible_answers = []
    subtask_desc1 = {"subtask_id": "subtask_1", "instruction": cot_sc_instruction, "agent_collaboration": "SC_CoT"}
    for agent in cot_agents:
        thinking, answer = await agent([taskInfo], cot_sc_instruction, is_sub_task=True)
        agents.append(f"CoT-SC agent {agent.id}, thinking: {thinking.content}; answer: {answer.content}")
        possible_thinkings.append(thinking)
        possible_answers.append(answer)
    final_decision_agent = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    synth_instr = "Sub-task 1: Synthesize the most consistent formal criteria definitions."  
    thinking1, answer1 = await final_decision_agent([taskInfo] + possible_thinkings + possible_answers, synth_instr, is_sub_task=True)
    sub_tasks.append(f"Sub-task 1 output: thinking - {thinking1.content}; answer - {answer1.content}")
    subtask_desc1['response'] = {"thinking": thinking1, "answer": answer1}
    logs.append(subtask_desc1)
    print("Step 1:", sub_tasks[-1])

    # Stage 2: Verify matrix properties
    # Sub-task 2: Unitarity of W (CoT)
    cot_instruction2 = "Sub-task 2: Compute W†W and verify if it equals identity to confirm W is unitary."  
    cot_agent2 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    thinking2, answer2 = await cot_agent2([taskInfo, thinking1, answer1], cot_instruction2, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent2.id}, thinking: {thinking2.content}; answer: {answer2.content}")
    sub_tasks.append(f"Sub-task 2 output: thinking - {thinking2.content}; answer - {answer2.content}")
    logs.append({"subtask_id": "subtask_2", "instruction": cot_instruction2, "response": {"thinking": thinking2, "answer": answer2}, "agent_collaboration": "CoT"})
    print("Step 2:", sub_tasks[-1])

    # Sub-task 3: Anti-Hermiticity of X and unitarity of e^X (SC_CoT)
    cot_sc_instruction3 = "Sub-task 3: Check X† = -X entrywise; then confirm (e^X)† e^X = I to test unitarity of e^X."  
    cot_agents3 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N)]
    possible_thinkings3 = []
    possible_answers3 = []
    for agent in cot_agents3:
        thinking, answer = await agent([taskInfo, thinking2, answer2], cot_sc_instruction3, is_sub_task=True)
        agents.append(f"CoT-SC agent {agent.id}, thinking: {thinking.content}; answer: {answer.content}")
        possible_thinkings3.append(thinking)
        possible_answers3.append(answer)
    final_decision_agent3 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    synth_instr3 = "Sub-task 3: Synthesize whether X is anti-Hermitian and whether e^X is unitary."  
    thinking3, answer3 = await final_decision_agent3([taskInfo, thinking2, answer2] + possible_thinkings3 + possible_answers3, synth_instr3, is_sub_task=True)
    agents.append(f"Final Decision Agent {final_decision_agent3.id}, thinking: {thinking3.content}; answer: {answer3.content}")
    sub_tasks.append(f"Sub-task 3 output: thinking - {thinking3.content}; answer - {answer3.content}")
    logs.append({"subtask_id": "subtask_3", "instruction": cot_sc_instruction3, "response": {"thinking": thinking3, "answer": answer3}, "agent_collaboration": "SC_CoT"})
    print("Step 3:", sub_tasks[-1])

    # Sub-task 4: Hermiticity of Z and non-Hermiticity of X (CoT)
    cot_instruction4 = "Sub-task 4: Compute Z† vs Z and X† vs X to determine which are Hermitian."  
    cot_agent4 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    thinking4, answer4 = await cot_agent4([taskInfo, thinking3, answer3], cot_instruction4, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent4.id}, thinking: {thinking4.content}; answer: {answer4.content}")
    sub_tasks.append(f"Sub-task 4 output: thinking - {thinking4.content}; answer - {answer4.content}")
    logs.append({"subtask_id": "subtask_4", "instruction": cot_instruction4, "response": {"thinking": thinking4, "answer": answer4}, "agent_collaboration": "CoT"})
    print("Step 4:", sub_tasks[-1])

    # Sub-task 5: Density matrix Y (SC_CoT)
    cot_sc_instruction5 = "Sub-task 5: Compute eigenvalues of Y to confirm they are ≥0 and check trace(Y)=1 to validate density matrix."  
    cot_agents5 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N)]
    possible_thinkings5 = []
    possible_answers5 = []
    for agent in cot_agents5:
        thinking, answer = await agent([taskInfo, thinking4, answer4], cot_sc_instruction5, is_sub_task=True)
        agents.append(f"CoT-SC agent {agent.id}, thinking: {thinking.content}; answer: {answer.content}")
        possible_thinkings5.append(thinking)
        possible_answers5.append(answer)
    final_decision_agent5 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    synth_instr5 = "Sub-task 5: Synthesize whether Y is positive semidefinite and trace-one."  
    thinking5, answer5 = await final_decision_agent5([taskInfo, thinking4, answer4] + possible_thinkings5 + possible_answers5, synth_instr5, is_sub_task=True)
    agents.append(f"Final Decision Agent {final_decision_agent5.id}, thinking: {thinking5.content}; answer: {answer5.content}")
    sub_tasks.append(f"Sub-task 5 output: thinking - {thinking5.content}; answer - {answer5.content}")
    logs.append({"subtask_id": "subtask_5", "instruction": cot_sc_instruction5, "response": {"thinking": thinking5, "answer": answer5}, "agent_collaboration": "SC_CoT"})
    print("Step 5:", sub_tasks[-1])

    # Stage 3: Debate on transformations
    # Sub-task 6: Norm change under e^X (Debate)
    debate_instr6 = (
        "Sub-task 6: Determine if there exists a vector whose Euclidean norm changes under multiplication by e^X."
        " Given solutions to the problem from other agents, consider their opinions as additional advice. Please think carefully and provide an updated answer."
    )
    debate_agents6 = [LLMAgentBase(["thinking", "answer"], "Debate Agent", model=self.node_model, role=role, temperature=0.5) for role in self.debate_role]
    all_think6 = []
    all_ans6 = []
    for agent in debate_agents6:
        thinking, answer = await agent([taskInfo, thinking3, answer3], debate_instr6, is_sub_task=True)
        agents.append(f"Debate agent {agent.id}, thinking: {thinking.content}; answer: {answer.content}")
        all_think6.append(thinking)
        all_ans6.append(answer)
    final_decision6 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    final6, ans6 = await final_decision6([taskInfo, thinking3, answer3] + all_think6 + all_ans6, "Sub-task 6: Given all above, reason carefully and conclude if norm can change.", is_sub_task=True)
    sub_tasks.append(f"Sub-task 6 output: thinking - {final6.content}; answer - {ans6.content}")
    logs.append({"subtask_id": "subtask_6", "instruction": debate_instr6, "response": {"thinking": final6, "answer": ans6}, "agent_collaboration": "Debate"})
    print("Step 6:", sub_tasks[-1])

    # Sub-task 7: Conjugation of Y by e^X (Debate)
    debate_instr7 = (
        "Sub-task 7: Compute (e^X)† Y (e^{-X}), check eigenvalues ≥0 and trace=1. "
        "Given solutions to the problem from other agents, consider their opinions as additional advice. Please think carefully and provide an updated answer."
    )
    debate_agents7 = [LLMAgentBase(["thinking", "answer"], "Debate Agent", model=self.node_model, role=role, temperature=0.5) for role in self.debate_role]
    all_think7 = []
    all_ans7 = []
    for agent in debate_agents7:
        thinking, answer = await agent([taskInfo, thinking5, answer5], debate_instr7, is_sub_task=True)
        agents.append(f"Debate agent {agent.id}, thinking: {thinking.content}; answer: {answer.content}")
        all_think7.append(thinking)
        all_ans7.append(answer)
    final_decision7 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    final7, ans7 = await final_decision7([taskInfo, thinking5, answer5] + all_think7 + all_ans7, "Sub-task 7: Given all above, reason carefully and conclude if transformed Y remains a quantum state.", is_sub_task=True)
    sub_tasks.append(f"Sub-task 7 output: thinking - {final7.content}; answer - {ans7.content}")
    logs.append({"subtask_id": "subtask_7", "instruction": debate_instr7, "response": {"thinking": final7, "answer": ans7}, "agent_collaboration": "Debate"})
    print("Step 7:", sub_tasks[-1])

    # Stage 4: Evaluate choices and synthesize
    # Sub-task 8: Choice 1 (Debate)
    debate_instr8 = (
        "Sub-task 8: Assess Choice 1: 'W and X represent evolution operators'."
        " Given solutions to the problem from other agents, consider their opinions as additional advice. Please think carefully and provide an updated answer."
    )
    debate_agents8 = [LLMAgentBase(["thinking", "answer"], "Debate Agent", model=self.node_model, role=role, temperature=0.5) for role in self.debate_role]
    all_think8 = []
    all_ans8 = []
    for agent in debate_agents8:
        thinking, answer = await agent([taskInfo, thinking2, answer2, thinking3, answer3], debate_instr8, is_sub_task=True)
        agents.append(f"Debate agent {agent.id}, thinking: {thinking.content}; answer: {answer.content}")
        all_think8.append(thinking)
        all_ans8.append(answer)
    final8_agent = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    final8, ans8 = await final8_agent([taskInfo, thinking2, answer2, thinking3, answer3] + all_think8 + all_ans8, "Sub-task 8: Given all above, conclude if Choice 1 is correct.", is_sub_task=True)
    sub_tasks.append(f"Sub-task 8 output: thinking - {final8.content}; answer - {ans8.content}")
    logs.append({"subtask_id": "subtask_8", "instruction": debate_instr8, "response": {"thinking": final8, "answer": ans8}, "agent_collaboration": "Debate"})
    print("Step 8:", sub_tasks[-1])

    # Sub-task 9: Choice 2 (Debate)
    debate_instr9 = (
        "Sub-task 9: Assess Choice 2: 'There exists a vector to which if one multiplies e^X, the norm changes'."
        " Given solutions to the problem from other agents, consider their opinions as additional advice. Please think carefully and provide an updated answer."
    )
    debate_agents9 = [LLMAgentBase(["thinking", "answer"], "Debate Agent", model=self.node_model, role=role, temperature=0.5) for role in self.debate_role]
    all_think9 = []
    all_ans9 = []
    for agent in debate_agents9:
        thinking, answer = await agent([taskInfo, final6, ans6], debate_instr9, is_sub_task=True)
        agents.append(f"Debate agent {agent.id}, thinking: {thinking.content}; answer: {answer.content}")
        all_think9.append(thinking)
        all_ans9.append(answer)
    final9_agent = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    final9, ans9 = await final9_agent([taskInfo, final6, ans6] + all_think9 + all_ans9, "Sub-task 9: Given all above, conclude if Choice 2 is correct.", is_sub_task=True)
    sub_tasks.append(f"Sub-task 9 output: thinking - {final9.content}; answer - {ans9.content}")
    logs.append({"subtask_id": "subtask_9", "instruction": debate_instr9, "response": {"thinking": final9, "answer": ans9}, "agent_collaboration": "Debate"})
    print("Step 9:", sub_tasks[-1])

    # Sub-task 10: Choice 3 (Debate)
    debate_instr10 = (
        "Sub-task 10: Assess Choice 3: '(e^X)* Y (e^{-X}) represents a quantum state'."
        " Given solutions to the problem from other agents, consider their opinions as additional advice. Please think carefully and provide an updated answer."
    )
    debate_agents10 = [LLMAgentBase(["thinking", "answer"], "Debate Agent", model=self.node_model, role=role, temperature=0.5) for role in self.debate_role]
    all_think10 = []
    all_ans10 = []
    for agent in debate_agents10:
        thinking, answer = await agent([taskInfo, final7, ans7], debate_instr10, is_sub_task=True)
        agents.append(f"Debate agent {agent.id}, thinking: {thinking.content}; answer: {answer.content}")
        all_think10.append(thinking)
        all_ans10.append(answer)
    final10_agent = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    final10, ans10 = await final10_agent([taskInfo, final7, ans7] + all_think10 + all_ans10, "Sub-task 10: Given all above, conclude if Choice 3 is correct.", is_sub_task=True)
    sub_tasks.append(f"Sub-task 10 output: thinking - {final10.content}; answer - {ans10.content}")
    logs.append({"subtask_id": "subtask_10", "instruction": debate_instr10, "response": {"thinking": final10, "answer": ans10}, "agent_collaboration": "Debate"})
    print("Step 10:", sub_tasks[-1])

    # Sub-task 11: Choice 4 (Debate)
    debate_instr11 = (
        "Sub-task 11: Assess Choice 4: 'Z and X represent observables'."
        " Given solutions to the problem from other agents, consider their opinions as additional advice. Please think carefully and provide an updated answer."
    )
    debate_agents11 = [LLMAgentBase(["thinking", "answer"], "Debate Agent", model=self.node_model, role=role, temperature=0.5) for role in self.debate_role]
    all_think11 = []
    all_ans11 = []
    for agent in debate_agents11:
        thinking, answer = await agent([taskInfo, thinking4, answer4], debate_instr11, is_sub_task=True)
        agents.append(f"Debate agent {agent.id}, thinking: {thinking.content}; answer: {answer.content}")
        all_think11.append(thinking)
        all_ans11.append(answer)
    final11_agent = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    final11, ans11 = await final11_agent([taskInfo, thinking4, answer4] + all_think11 + all_ans11, "Sub-task 11: Given all above, conclude if Choice 4 is correct.", is_sub_task=True)
    sub_tasks.append(f"Sub-task 11 output: thinking - {final11.content}; answer - {ans11.content}")
    logs.append({"subtask_id": "subtask_11", "instruction": debate_instr11, "response": {"thinking": final11, "answer": ans11}, "agent_collaboration": "Debate"})
    print("Step 11:", sub_tasks[-1])

    # Sub-task 12: Synthesize final answer (CoT)
    cot_instruction12 = "Sub-task 12: Based on the assessments of all four choices, select and justify the correct statement."  
    cot_agent12 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    thinking12, answer12 = await cot_agent12([
        taskInfo, final8, ans8, final9, ans9, final10, ans10, final11, ans11
    ], cot_instruction12, is_sub_task=True)
    sub_tasks.append(f"Sub-task 12 output: thinking - {thinking12.content}; answer - {answer12.content}")
    logs.append({"subtask_id": "subtask_12", "instruction": cot_instruction12, "response": {"thinking": thinking12, "answer": answer12}, "agent_collaboration": "CoT"})
    print("Step 12:", sub_tasks[-1])

    final_answer = await self.make_final_answer(thinking12, answer12, sub_tasks, agents)
    return final_answer, logs