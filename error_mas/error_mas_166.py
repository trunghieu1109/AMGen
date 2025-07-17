import asyncio
from collections import Counter

async def forward_166(self, taskInfo):
    sub_tasks = []
    agents = []
    logs = []

    # Sub-task 1: Derive normalized state and density matrix (SC_CoT)
    cot_sc_instruction = "Sub-task 1: Derive the normalized Schrödinger cat state |psi> and its density matrix rho for general phi and alpha, expressing N, cos(phi), sin(phi), and overlap e^(-2alpha^2)."
    N_sc = self.max_sc
    cot_agents1 = [LLMAgentBase(["thinking","answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N_sc)]
    possible_answers1 = []
    possible_thinkings1 = []
    for agent in cot_agents1:
        thinking, answer = await agent([taskInfo], cot_sc_instruction, is_sub_task=True)
        agents.append(f"CoT-SC agent {agent.id}, thinking: {thinking.content}; answer: {answer.content}")
        possible_thinkings1.append(thinking)
        possible_answers1.append(answer)
    final_decision_agent1 = LLMAgentBase(["thinking","answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking1, answer1 = await final_decision_agent1([taskInfo] + possible_thinkings1 + possible_answers1, "Sub-task 1: Synthesize and choose the most consistent derivation for state and density matrix.", is_sub_task=True)
    sub_tasks.append(f"Sub-task 1 output: thinking - {thinking1.content}; answer - {answer1.content}")
    agents.append(f"Final Decision agent for subtask_1, thinking: {thinking1.content}; answer: {answer1.content}")
    logs.append({"subtask_id":"subtask_1","instruction":cot_sc_instruction,"context":["user query"],"agent_collaboration":"SC_CoT","response":{"thinking":thinking1,"answer":answer1})
    print("Step 1: ", sub_tasks[-1])

    # Sub-task 2: Compute moments and define tau (Debate)
    debate_instruction = "Sub-task 2: Compute the first and second moments (mean displacement and covariance) of rho and define the reference Gaussian state tau that matches these moments. Given solutions to the problem from other agents, consider their opinions as additional advice. Please think carefully and provide an updated answer."
    debate_agents2 = [LLMAgentBase(["thinking","answer"], "Debate Agent", model=self.node_model, role=role, temperature=0.5) for role in self.debate_role]
    N_round2 = self.max_round
    all_thinking2 = [[] for _ in range(N_round2)]
    all_answer2 = [[] for _ in range(N_round2)]
    for r in range(N_round2):
        for i, agent in enumerate(debate_agents2):
            if r == 0:
                thinking2, answer2 = await agent([taskInfo, thinking1, answer1], debate_instruction, r, is_sub_task=True)
            else:
                prev = all_thinking2[r-1] + all_answer2[r-1]
                thinking2, answer2 = await agent([taskInfo, thinking1, answer1] + prev, debate_instruction, r, is_sub_task=True)
            agents.append(f"Debate agent {agent.id}, round {r}, thinking: {thinking2.content}; answer: {answer2.content}")
            all_thinking2[r].append(thinking2)
            all_answer2[r].append(answer2)
    final_decision_agent2 = LLMAgentBase(["thinking","answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking2_f, answer2_f = await final_decision_agent2([taskInfo, thinking1, answer1] + all_thinking2[-1] + all_answer2[-1], "Sub-task 2: Given all the above thinking and answers, reason over them carefully and provide a final definition of tau.", is_sub_task=True)
    sub_tasks.append(f"Sub-task 2 output: thinking - {thinking2_f.content}; answer - {answer2_f.content}")
    agents.append(f"Final Decision agent for subtask_2, thinking: {thinking2_f.content}; answer: {answer2_f.content}")
    logs.append({"subtask_id":"subtask_2","instruction":debate_instruction,"context":["user query","output of subtask_1"],"agent_collaboration":"Debate","response":{"thinking":thinking2_f,"answer":answer2_f})
    print("Step 2: ", sub_tasks[-1])

    # Sub-task 3: Compute entropic terms symbolically (SC_CoT)
    cot_sc_instruction3 = "Sub-task 3: Compute the entropic terms Tr[rho ln rho] and Tr[tau ln tau] symbolically, deriving delta_nG = Tr[rho ln rho] - Tr[tau ln tau]."
    N_sc3 = self.max_sc
    cot_agents3 = [LLMAgentBase(["thinking","answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N_sc3)]
    possible_thinkings3 = []
    possible_answers3 = []
    for agent in cot_agents3:
        thinking3_i, answer3_i = await agent([taskInfo, thinking1, answer1, thinking2_f, answer2_f], cot_sc_instruction3, is_sub_task=True)
        agents.append(f"CoT-SC agent {agent.id}, thinking: {thinking3_i.content}; answer: {answer3_i.content}")
        possible_thinkings3.append(thinking3_i)
        possible_answers3.append(answer3_i)
    final_decision_agent3 = LLMAgentBase(["thinking","answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking3, answer3 = await final_decision_agent3([taskInfo, thinking1, answer1, thinking2_f, answer2_f] + possible_thinkings3 + possible_answers3, "Sub-task 3: Synthesize and choose the most consistent symbolic entropy derivation.", is_sub_task=True)
    sub_tasks.append(f"Sub-task 3 output: thinking - {thinking3.content}; answer - {answer3.content}")
    agents.append(f"Final Decision agent for subtask_3, thinking: {thinking3.content}; answer: {answer3.content}")
    logs.append({"subtask_id":"subtask_3","instruction":cot_sc_instruction3,"context":["user query","output of subtask_2"],"agent_collaboration":"SC_CoT","response":{"thinking":thinking3,"answer":answer3})
    print("Step 3: ", sub_tasks[-1])

    # Sub-task 4: Numeric substitution and finalize (Reflexion)
    reflect_inst = "Given previous attempts and feedback, carefully consider where you could go wrong in your latest attempt. Using insights from previous attempts, try to solve the task better."
    cot_reflect_instruction = "Sub-task 4: Substitute phi = -pi/4 and alpha = 0.5 into the delta_nG expression, calculate its numerical value, compare to the four options [2.48, 0, 1.38, 0.25], and select the correct answer." + reflect_inst
    cot_agent4 = LLMAgentBase(["thinking","answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    critic_agent4 = LLMAgentBase(["feedback","correct"], "Critic Agent", model=self.node_model, temperature=0.0)
    cot_inputs = [taskInfo, thinking3, answer3]
    thinking4, answer4 = await cot_agent4(cot_inputs, cot_reflect_instruction, 0, is_sub_task=True)
    agents.append(f"Reflexion CoT agent {cot_agent4.id}, thinking: {thinking4.content}; answer: {answer4.content}")
    for i in range(self.max_round):
        critic_inst = "Please review the answer above and criticize on where might be wrong. If you are absolutely sure it is correct, output exactly 'True' in 'correct'"
        feedback, correct = await critic_agent4([taskInfo, thinking4, answer4], critic_inst, i, is_sub_task=True)
        agents.append(f"Critic agent {critic_agent4.id}, feedback: {feedback.content}; correct: {correct.content}")
        if correct.content == "True":
            break
        cot_inputs.extend([thinking4, answer4, feedback])
        thinking4, answer4 = await cot_agent4(cot_inputs, cot_reflect_instruction, i+1, is_sub_task=True)
        agents.append(f"Reflexion CoT agent {cot_agent4.id}, refined thinking: {thinking4.content}; refined answer: {answer4.content}")
    sub_tasks.append(f"Sub-task 4 output: thinking - {thinking4.content}; answer - {answer4.content}")
    logs.append({"subtask_id":"subtask_4","instruction":cot_reflect_instruction,"context":["user query","output of subtask_3"],"agent_collaboration":"Reflexion","response":{"thinking":thinking4,"answer":answer4})
    print("Step 4: ", sub_tasks[-1])

    final_answer = await self.make_final_answer(thinking4, answer4, sub_tasks, agents)
    return final_answer, logs